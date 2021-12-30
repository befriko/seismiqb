""" A holder for horizon extension steps inherited from `.class:Enhancer` with:
    - redifined get_mask_transform_ppl to thin out loaded mask
    - making an iterative inference to cover the holes in a given horizon.
"""
import gc
from copy import copy
from pprint import pformat
from time import perf_counter

import numpy as np
import torch

from .enhancer import Enhancer

from ..labels import Horizon
from ..samplers import ExtensionGrid

from ...batchflow import Pipeline, Config, Notifier, Monitor
from ...batchflow import B, F, C, V, P, R



class Extender(Enhancer):
    """ Provides interface for train, inference and quality assesment for the task of horizon extension. """
    #pylint: disable=attribute-defined-outside-init
    DEFAULTS = Config({
        **Enhancer.DEFAULTS,
        'inference': {
            'batch_size': None,
            'crop_shape': None,
            'width': 3,

            'n_steps': 100,
            'stride': 32,
            'threshold': 10,

            # Merge parameters
            'merge' : {'mean_threshold' : 0.5, 'max_threshold': 1.5, 'adjacency': 0},

            'prefetch': 0,
        }
    })

    def inference(self, horizon, model, config=None, **kwargs):
        """ Fill the holes of a given horizon with the supplied model.

        Works by making predictions near the horizon boundaries and stitching them to the original one.
        """
        # Prepare parameters
        config = config or {}
        config = Config({**self.config['common'], **self.config['inference'], **config, **kwargs})
        n_steps, stride, batch_size, crop_shape = config.get(['n_steps', 'stride', 'batch_size', 'crop_shape'])
        merge_params = config.get('merge')
        threshold = config.get('threshold', 25)
        prefetch = config.get('prefetch', 0)

        # Log: pipeline_config to a file
        self.log_to_file(pformat(config.config, depth=2), '末 inference_config.txt')

        # Start resource tracking
        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.5, gpu_list=self.gpu_list)
            monitor.__enter__()

        # Make dataset and copy horizon
        horizon = copy(horizon)
        dataset = self.make_dataset(horizon=horizon)

        prev_len = initial_len = len(horizon)
        self.log(f'Inference started for {n_steps} steps with stride {stride}')
        notifier = Notifier(self.config.bar,
                            desc='Extender inference',
                            file=self.make_savepath('末 inference.log'))

        start_time = perf_counter()
        for i in notifier(range(n_steps)):
            # Create grid of crops near horizon holes
            grid = ExtensionGrid(horizon=horizon, crop_shape=crop_shape,
                                 batch_size=batch_size, stride=stride,
                                 top=1, threshold=5)
            config['grid'] = grid

            # Create pipeline TODO: make better `add_model`
            inference_pipeline = self.get_inference_template() << config << dataset
            inference_pipeline.models.add_model('model', model)
            inference_pipeline.run(n_iters=grid.n_iters, prefetch=prefetch)

            # Merge surfaces on crops to the horizon itself
            horizon_patches = inference_pipeline.v('predicted_horizons')
            horizon, _, _ = Horizon.merge_into(horizon, horizon_patches, **merge_params)

            # Log length increase
            curr_len = len(horizon)
            if (curr_len - prev_len) < threshold:
                break
            self.log(f'Iteration {i}:  extended from {prev_len} to {curr_len}, + {curr_len - prev_len}')
            prev_len = curr_len

            # Cleanup
            gc.collect()
            inference_pipeline.reset('variables')

        notifier.close()
        elapsed = perf_counter() - start_time

        # Log: resource graphs
        if self.monitor:
            monitor.__exit__(None, None, None)
            monitor.visualize(savepath=self.make_savepath('末 inference_resource.png'), show=self.plot)

        self.log(f'Total points added: {curr_len - initial_len}')

        torch.cuda.empty_cache()
        horizon.name = f'extended_{horizon.name[8:]}' # get rid of `copy_of_` postfix

        self.inference_log = {
            'elapsed': elapsed,
            'added_points': curr_len - initial_len,
        }
        return horizon


    def distortion_pipeline(self):
        """ Defines transformations performed with `masks` component. """
        def functor(scale):
            return lambda m: np.sin(m[:, 0] * scale)

        return (
            Pipeline()
            .transpose(src='masks', order=(1, 2, 0))
            .filter_out(src='masks', dst='prior_masks',
                        expr=lambda m: m[:, 0],
                        low=P(R('uniform', low=0., high=0.4)),
                        length=P(R('uniform', low=0.30, high=0.5)))
            .filter_out(src='prior_masks', dst='prior_masks',
                        expr=F(functor)(R('uniform', low=15, high=35)), low=0.0, p=0.7)
            .transpose(src=['masks', 'prior_masks'], order=(2, 0, 1))
        )


    def get_inference_template(self):
        """ Defines inference pipeline. """
        inference_template = (
            Pipeline()

            # Init everything
            .init_variable('predicted_horizons', default=[])

            # Load data
            .make_locations(generator=C('grid'))
            .load_cubes(dst='images')
            .create_masks(dst='prior_masks', width=C('width', default=3))
            .adaptive_reshape(src=['images', 'prior_masks'])
            .normalize(src='images')
            .concat_components(src=['images', 'prior_masks'], dst='images', axis=1)

            # Use model for prediction
            .predict_model('model',
                           B('images'),
                           fetches='predictions',
                           save_to=B('predicted_masks', mode='w'))
            .masks_to_horizons(src_masks='predicted_masks', threshold=0.5, minsize=16,
                               dst='horizons', skip_merge=True)
            .update(V('predicted_horizons', mode='e'), B('horizons'))
        )
        return inference_template


    # One method to rule them all
    def run(self, cube_paths=None, horizon_paths=None, horizon=None, **kwargs):
        """ Run the entire procedure of horizon extension. """
        dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)
        horizon = dataset.labels[0][0]

        model = self.train(horizon=horizon, **kwargs)

        prediction = self.inference(horizon, model, **kwargs)
        prediction = self.postprocess(prediction)
        info = self.evaluate(prediction, dataset=dataset)
        return prediction, info
