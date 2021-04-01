""" A convenient holder for faults detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
"""
import os
import glob
import datetime

import numpy as np
import torch

from ...batchflow import Config, Pipeline
from ...batchflow import B, C, D, P, R, V, F
from ...batchflow.models.torch import TorchModel, ResBlock, EncoderDecoder
from .base import BaseController
from ..cubeset import SeismicCubeset
from ..fault import Fault
from ..layers import InputLayer
from ..utils import adjust_shape_3d, fill_defaults

class FaultController(BaseController):
    DEFAULTS = Config({
        **BaseController.DEFAULTS,
        # Data
        'dataset': {
            'path': '/cubes/',
            'train_cubes': [],
            'transposed_cubes': [],
            'label_dir': '/INPUTS/FAULTS/NPY_WIDTH_{}/*',
            'width': 3,
        },

        # Model parameters
        'train': {
            # Augmentation parameters
            'batch_size': 1024,
            'microbatch': 8,
            'side_view': False,
            'angle': 25,
            'scale': (0.7, 1.5),
            'crop_shape': [1, 128, 512],
            'filters': [64, 96, 128, 192, 256],
            'itemwise': True,
            'phase': True,
            'continuous_phase': False,
            'model': 'UNet',
            'loss': 'bce',
            'output': 'sigmoid',
            'slicing': 'native',
        },

        'inference': {
            'cubes': {
                '21_AYA': [],
            },
            'batch_size': 32,
            'side_view': False,
            'crop_shape': [1, 128, 512],
            'inference_batch_size': 32,
            'inference_chunk_shape': (100, None, None),
            'smooth_borders': False,
            'stride': 0.5,
            'orientation': 'ilines',
            'slicing': 'native',
            'output': 'sigmoid',
            'itemwise': True
        }
    })
    # .run_later(D('size'), n_iters=C('n_iters'), n_epochs=None, prefetch=0, profile=False, bar=C('bar')

    BASE_MODEL_CONFIG = {
        'optimizer': {'name': 'Adam', 'lr': 0.01},
        "decay": {'name': 'exp', 'gamma': 0.9, 'frequency': 100, 'last_iter': 2000},
        'microbatch': C('microbatch'),
        'initial_block': {
            'enable': C('phase'),
            'filters': C('filters')[0] // 2,
            'kernel_size': 5,
            'downsample': False,
            'attention': 'scse',
            'phases': C('phase'),
            'continuous': C('continuous_phase')
        },
        'loss': C('loss')
    }

    UNET_CONFIG = {
        'initial_block/base_block': InputLayer,
        'body/encoder': {
            'num_stages': 4,
            'order': 'sbd',
            'blocks': {
                'base': ResBlock,
                'n_reps': 1,
                'filters': C('filters')[:-1],
                'attention': 'scse',
            },
        },
        'body/embedding': {
            'base': ResBlock,
            'n_reps': 1,
            'filters': C('filters')[-1],
            'attention': 'scse',
        },
        'body/decoder': {
            'num_stages': 4,
            'upsample': {
                'layout': 'tna',
                'kernel_size': 5,
            },
            'blocks': {
                'base': ResBlock,
                'filters': C('filters')[-2::-1],
                'attention': 'scse',
            },
        },
        'head': {
            'base_block': ResBlock,
            'filters': [16, 8],
            'attention': 'scse'
        },
        'output': torch.sigmoid,
        'common/activation': 'relu6',
        'loss': C('loss')
    }

    def make_dataset(self, ratios=None, **kwargs):
        config = {**self.config['dataset'], **kwargs}
        width = config['width']
        label_dir = config['label_dir']
        paths = [self.amplitudes_path(item) for item in config['train_cubes']]

        dataset = SeismicCubeset(index=paths)
        dataset.load(label_dir=label_dir.format(width), labels_class=Fault, transform=True, verify=True)
        dataset.modify_sampler(dst='train_sampler', finish=True, low=0.0, high=1.0)

        if ratios is None:
            ratios = {}

            if len(dataset) > 0:
                for i in range(len(dataset)):
                    faults = dataset.labels[i]
                    fault_area = sum([len(np.unique(faults[j].points)) for j in range(len(faults))])
                    cube_area = np.prod(dataset.geometries[i].cube_shape)
                    ratios[dataset.indices[i]] = fault_area / cube_area
            else:
                ratios[dataset.indices[0]] = 1

        weights = np.array([ratios[i] for i in dataset.indices])
        weights /= weights.sum()
        weights = weights.clip(max=0.3)
        weights = weights.clip(min=0.1)
        weights /= weights.sum()

        dataset.create_sampler(p=list(weights))
        dataset.modify_sampler(dst='train_sampler', finish=True, low=0.0, high=1.0)

        return dataset

    def load_pipeline(self, create_masks=True, train=True, **kwargs):
        """ Create loading pipeline common for train and inference stages.

        Parameters
        ----------
        create_masks : bool, optional
            create mask or not, by default True
        use_adjusted_shapes : bool, optional
            use or not adjusted shapes to perform augmentations changing shape (rotations and scaling),
            by default False.

        Returns
        -------
        batchflow.Pipeline
        """
        load_shape = F(np.array)(F(self.adjust_shape)(C('crop_shape'), C('angle'), C('scale')[0])) if train else C('crop_shape')
        shape = {self.cube_name_from_alias(k): load_shape for k in self.config['dataset/train_cubes']}
        shape.update({self.cube_name_from_alias(k): load_shape[[1, 0, 2]] for k in self.config['dataset/transposed_cubes']})

        if train:
            ppl = Pipeline().make_locations(points=D('train_sampler')(C('batch_size')), shape=shape, side_view=C('side_view'))
        else:
            ppl = Pipeline().make_locations(points=D('grid_gen')(), shape=C('test_crop_shape'))

        ppl += Pipeline().load_cubes(dst='images', slicing=C('slicing'))

        if create_masks:
            ppl +=  Pipeline().create_masks(dst='masks')
            components = ['images', 'masks']
        else:
            components = ['images']

        ppl += (Pipeline()
            .adaptive_reshape(src=components, shape=load_shape)
            .normalize(mode='q', itemwise=C('itemwise'), src='images')
        )
        return ppl

    def augmentation_pipeline(self, **kwargs):
        return (Pipeline()
            .transpose(src=['images', 'masks'], order=(1, 2, 0))
            .flip(axis=1, src=['images', 'masks'], seed=P(R('uniform', 0, 1)), p=0.3)
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .rotate(angle=P(R('uniform', -C('angle'), C('angle'))), src=['images', 'masks'], p=0.3)
            .scale_2d(scale=P(R('uniform', C('scale')[0], C('scale')[1])), src=['images', 'masks'], p=0.3)
            .transpose(src=['images', 'masks'], order=(2, 0, 1))
            .central_crop(C('crop_shape'), src=['images', 'masks'])
            .cutout_2d(src=['images', 'masks'], patch_shape=np.array((1, 40, 40)), n=3, p=0.2)
        )

    def train_pipeline(self, **kwargs):
        model_class = F(self.get_model_class)(C('model'))
        model_config = F(self.get_model_config)(C('model'))
        return (Pipeline()
            .init_variable('loss_history', [])
            .init_model('dynamic', model_class, 'model', model_config)
            .add_channels(src=['images', 'masks'])
            .train_model('model',
                         fetches=['loss', C('output')],
                         images=B('images'),
                         masks=B('masks'),
                         save_to=[V('loss_history', mode='w'), B('predictions')])
        )

    def get_train_template(self, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        return (
            self.load_pipeline(create_masks=True, train=True, **kwargs) +
            self.augmentation_pipeline(**kwargs) +
            self.train_pipeline(**kwargs)
        )

    def get_inference_template(self, train_pipeline=None, model_path=None, create_masks=False):
        if train_pipeline is not None:
            test_pipeline = Pipeline().import_model('model', train_pipeline)
        else:
            test_pipeline = Pipeline().load_model(mode='dynamic', model_class=TorchModel, name='model', path=model_path)

        test_pipeline += self.load_pipeline(create_masks=create_masks, train=False)

        if create_masks:
            comp = ['images', 'masks']
        else:
            comp = ['images']

        test_pipeline += (
            Pipeline()
            .adaptive_reshape(src=comp, shape=C('crop_shape'))
            .add_channels(src=comp)
            .init_variable('predictions', [])
            .init_variable('target', [])
            .predict_model('model', B('images'), fetches=C('output'), save_to=B('predictions'))
            .run_later(D('size'))
        )

        smooth_borders = self.config['inference/smooth_borders']
        if smooth_borders:
            if isinstance(smooth_borders, bool):
                step = 0.1
            else:
                step = smooth_borders
            test_pipeline += Pipeline().update(B('predictions') , F(self.smooth_borders)(B('predictions'), step))

        if create_masks:
            test_pipeline += Pipeline().update(V('target', mode='e'), B('masks'))
        test_pipeline += Pipeline().update(V('predictions', mode='e'), B('predictions'))
        return test_pipeline

    def make_inference_dataset(self, labels=False, **kwargs):
        config = {**self.config['inference'], **self.config['dataset'], **kwargs}
        inference_cubes = config['cubes']
        width = config['width']
        label_dir = config['label_dir']

        cubes_paths = [self.amplitudes_path(item) for item in inference_cubes]
        dataset = SeismicCubeset(index=cubes_paths)
        if labels:
            dataset.load(label_dir=label_dir.format(width), labels_class=Fault, transform=True, verify=True, bar=False)
        else:
            dataset.load_geometries()
        return dataset

    def parse_locations(self, cubes):
        cubes = cubes.copy()
        if isinstance(cubes, (list, tuple)):
            cubes = {cube: (0, None, None, None) for cube in cubes}
        for cube in cubes:
            cubes[cube] = [cubes[cube]] if isinstance(cubes[cube], (list, tuple)) else cubes[cube]
        return cubes

    def inference_on_slides(self, train_pipeline=None, model_path=None, create_mask=False, **kwargs):
        config = {**self.config['inference'], **kwargs}
        strides = config['stride'] if isinstance(config['stride'], tuple) else [config['stride']] * 3
        batch_size = config['batch_size']

        dataset = self.make_inference_dataset(create_mask)
        inference_pipeline = self.get_inference_template(train_pipeline, model_path, create_mask)
        inference_pipeline.set_config(config)

        inference_cubes = {
            self.cube_name_from_path(self.amplitudes_path(k)): v for k, v in self.parse_locations(config['cubes']).items()
        }

        outputs = {}
        for cube_idx in dataset.indices:
            outputs[cube_idx] = []
            geometry = dataset.geometries[cube_idx]
            shape = geometry.cube_shape
            for item in inference_cubes[cube_idx]:
                axis = item[0]
                slices = item[1:]
                if axis in [0, 'i', 'ilines']:
                    crop_shape = config['crop_shape']
                    order = (0, 1, 2)
                else:
                    crop_shape = np.array(config['crop_shape'])[[1, 0, 2]]
                    order = (1, 0, 2)
                inference_pipeline.set_config({'test_crop_shape': crop_shape})
                _strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)
                slices = fill_defaults(slices, [[0, i] for i in shape])

                dataset.make_grid(cube_idx, crop_shape, *slices, strides=_strides, batch_size=batch_size)

                ppl = (inference_pipeline << dataset)
                for _ in range(dataset.grid_iters):
                    _ = ppl.next_batch(D('size'))
                prediction = dataset.assemble_crops(ppl.v('predictions'), order=order).astype('float32')
                image = geometry.file_hdf5['cube'][
                    slices[0][0]:slices[0][1],
                    slices[1][0]:slices[1][1],
                    slices[2][0]:slices[2][1]
                ]
                outputs[cube_idx] += [[image, prediction]]
                if create_mask:
                    outputs[cube_idx][-1] += dataset.assemble_crops(ppl.v('target'), order=order).astype('float32')
        return outputs

    def get_model_config(self, name):
        if name == 'UNet':
            return {**self.BASE_MODEL_CONFIG, **self.UNET_CONFIG}
        raise ValueError(f'Unknown model name: {name}')

    def get_model_class(self, name):
        if name == 'UNet':
            return EncoderDecoder
        return TorchModel

    def amplitudes_path(self, cube):
        return glob.glob(self.config['dataset/path'] + 'CUBE_' + cube + '/amplitudes*.hdf5')[0]

    def cube_name_from_alias(self, path):
        return os.path.splitext(self.amplitudes_path(path).split('/')[-1])[0]

    def cube_name_from_path(self, path):
        return os.path.splitext(path.split('/')[-1])[0]

    def create_filename(self, prefix, orientation, ext):
        return (prefix + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_{}.{}').format(orientation, ext)

    @classmethod
    def adjust_shape(cls, crop_shape, angle, scale):
        crop_shape = np.array(crop_shape)
        load_shape = adjust_shape_3d(crop_shape[[1, 2, 0]], angle, scale=scale)
        return (load_shape[2], load_shape[0], load_shape[1])

    def smooth_borders(self, crops, step):
        mask = self.border_smoothing_mask(crops.shape[-3:], step)
        mask = np.expand_dims(mask, axis=0)
        if len(crops.shape) == 5:
            mask = np.expand_dims(mask, axis=0)
        crops = crops * mask
        return crops

    def border_smoothing_mask(self, shape, step):
        mask = np.ones(shape)
        axes = [(1, 2), (0, 2), (0, 1)]
        if isinstance(step, (int, float)):
            step = [step] * 3
        for i in range(len(step)):
            if isinstance(step[i], float):
                step[i] = int(shape[i] * step[i])
            length = shape[i]
            if length >= 2 * step[i]:
                _mask = np.ones(length, dtype='float32')
                _mask[:step[i]] = np.linspace(0, 1, step[i]+1)[1:]
                _mask[:-step[i]-1:-1] = np.linspace(0, 1, step[i]+1)[1:]
                _mask = np.expand_dims(_mask, axes[i])
                mask = mask * _mask
        return mask