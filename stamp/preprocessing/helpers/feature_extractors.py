import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import PIL
import numpy as np
#no marugoto dependency
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import json
import h5py
import uni
import os
from collections.abc import Iterable
import timm
import itertools
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from conch.open_clip_custom import create_model_from_pretrained
from typing import TypeVar, Callable
import sys

from .swin_transformer import swin_tiny_patch4_window7_224, ConvStem

__version__ = "001_01-10-2023"

def get_digest(file: str):
    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

class FeatureExtractorCTP:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles.
        """
        digest = get_digest(self.checkpoint_path)
        assert digest == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

        self.model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()

        ctranspath = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ctranspath['model'], strict=True)
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name='xiyuewang-ctranspath-7c998680'

        print("CTransPath model successfully initialised...\n")
        return model_name
        
class FeatureExtractorChiefCTP:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles.
        """
        #Architecture is identical, only the weights differ from CTP
        self.model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()

        ctranspath = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ctranspath['model'], strict=True)
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name='chief-ctp'

        print("ChiefCTP model successfully initialised...\n")
        return model_name
    
class FeatureExtractorUNI:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles. 
        Requirements: 
            Permission from authors via huggingface: https://huggingface.co/MahmoodLab/UNI
            Huggingface account with valid login token
        On first model initialization, you will be prompted to enter your login token. The token is
        then stored in ./home/<user>/.cache/huggingface/token. Subsequent inits do not require you to re-enter the token. 

        Args:
            device: "cuda" or "cpu"
        """
        asset_dir = f"{os.environ['STAMP_RESOURCES_DIR']}/uni"
        model, transform = uni.get_encoder(enc_name="uni", device=device, assets_dir=asset_dir)
        self.model = model
        self.transform = transform

        digest = get_digest(f"{asset_dir}/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin")
        model_name = f"mahmood-uni-{digest[:8]}"

        print("UNI model successfully initialised...\n")
        return model_name
    

class FeatureExtractorProvGP:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using GigaPath tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'prov-gigapath'
        checkpoint = 'pytorch_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Load the model structure
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)

        # Load the state dict from the checkpoint file
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # Define the transform
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        print("GigaPath tile encoder model successfully initialized...\n")
        return model_name

class FeatureExtractorHibouB:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Hibou B tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'hibou-b'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        processor_path = os.path.join(ckpt_dir, 'processor')

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Load the model configuration
        config = AutoConfig.from_pretrained("histai/hibou-b", trust_remote_code=True)

        # Initialize the model with the config
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Load the saved model weights
        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816])
        ])

        print("Hibou-B model successfully initialised...\n")
        return model_name

class FeatureExtractorHibouL:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Hibou L tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'hibou-l'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        processor_path = os.path.join(ckpt_dir, 'processor')

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Load the model configuration
        config = AutoConfig.from_pretrained("histai/hibou-L", trust_remote_code=True)

        # Initialize the model with the config
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Load the saved model weights
        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816])
        ])

        print("Hibou-L model successfully initialised...\n")
        return model_name

class FeatureExtractorKaiko:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Kaiko tile encoder."""
        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"

        model_name = 'kaiko-vitl14'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        
        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitl14", trust_repo=True, pretrained=False)
        self.model.load_state_dict(torch.load(ckpt_path))

        from torchvision.transforms import v2
        # initialize the model pre-process transforms
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=224),
                v2.CenterCrop(size=224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        print("Kaiko model successfully initialised...\n")
        return model_name

class FeatureExtractorConch:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using CONCH tile encoder."""
        
        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'conch'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)

        # Initialize the model (you may need to pass other necessary parameters as per the create_model_from_pretrained function)
        self.model, self.processor = create_model_from_pretrained('conch_ViT-B-16', ckpt_path, force_image_size=224)

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        r"""OpenAI color normalization mean in RGB format (values in 0-1)."""
        r"""OpenAI color normalization std in RGB format (values in 0-1)."""
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        r"""ImageNet color normalization mean in RGB format (values in 0-1)."""
        r"""ImageNet color normalization std in RGB format (values in 0-1)."""
        IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

        from typing import Sequence
        mean: Sequence[float] = OPENAI_DATASET_MEAN
        std: Sequence[float] = OPENAI_DATASET_STD

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        print("CONCH model successfully initialised...\n")
        return model_name

class FeatureExtractorPhikon:
    def init_feat_extractor(self, device: str, weights_path: str = None, **kwargs):
        """Extracts features from slide tiles using Phikon tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'phikon'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        processor_path = os.path.join(ckpt_dir, 'processor')

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Load the model configuration
        config = AutoConfig.from_pretrained("owkin/phikon", trust_remote_code=True)

        # Initialize the model with the config
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Load the saved model weights
        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Phikon model successfully initialized...\n")
        return model_name

class FeatureExtractorVirchow:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Virchow tile encoder."""

        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'virchow'
        checkpoint = 'pytorch_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Load the model structure
        self.model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        # Load the state dict from the checkpoint file
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # Define the transform
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        
        print("Virchow model successfully initialized...\n")
        return model_name
    
class FeatureExtractorVirchow2:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Virchow2 tile encoder."""

        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'virchow2'
        checkpoint = 'pytorch_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Load the model structure
        self.model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        # Load the state dict from the checkpoint file
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # Define the transform
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        
        print("Virchow2 model successfully initialized...\n")
        return model_name

class FeatureExtractorHOptimus0:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using H-optimus-0 tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'hoptimus0'
        checkpoint = 'pytorch_model.bin'
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        self.model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=False, init_values=1e-5, dynamic_img_size=False)
        
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

        print("H-optimus-0 model successfully initialized...\n")
        return model_name
    

T = TypeVar("T")

class SlideTileDataset(Dataset[T]):
    def __init__(
        self,
        patches: np.ndarray,
        transform: Callable[[PIL.Image.Image], T],
        *,
        repetitions: int = 1
    ) -> None:
        self.tiles = patches
        #assert self.tiles, f'no tiles found in {slide_dir}'
        self.tiles *= repetitions
        self.transform = transform

    # patchify returns a NumPy array with shape (n_rows, n_cols, 1, H, W, N), if image is N-channels.
    # H W N is Height Width N-channels of the extracted patch
    # n_rows is the number of patches for each column and n_cols is the number of patches for each row
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i: int) -> T:
        image = PIL.Image.fromarray(self.tiles[i])
        image = self.transform(image)

        return image

def batched(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if batch_size < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def extract_features_(
        *,
        model: nn.Module, model_name, transform: Callable[[PIL.Image.Image], torch.Tensor],
        norm_wsi_img: np.ndarray, coords: list, wsi_name: str, outdir: Path,
        augmented_repetitions: int = 0, cores: int = 8, is_norm: bool = True, device: str = 'cpu',
        target_microns: int = 256, patch_size: int = 224, processor=None
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """

    extractor_string = f'STAMP-extract-{__version__}_{model_name}'
    with open(outdir.parent/'info.json', 'w') as f:
        json.dump({'extractor': extractor_string,
                  'augmented_repetitions': augmented_repetitions,
                  'normalized': is_norm,
                  'microns': target_microns,
                  'patch_size': patch_size}, f)

    unaugmented_ds: Dataset[PIL.Image.Image] = SlideTileDataset(norm_wsi_img, transform=lambda x: x)
    augmented_ds = []

    #clean up memory
    del norm_wsi_img

    ds: Dataset[PIL.Image.Image] = ConcatDataset([unaugmented_ds, augmented_ds])

    dl = torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=False, num_workers=cores, drop_last=False, pin_memory=(device != 'cpu'))


    # We do this because we can't put PIL images into a tensor
    # / would have to transform back and forth a buch of times.
    # FIXME: Rewrite all of this so we unify all the preprocessing
    # following e.g. `transformers.image_processing_utils.BaseImageProcessor`

    from math import ceil

    batch_size = 64

    batched_dl = batched(dl, batch_size=batch_size)

    batched_dl: Iterable[list[PIL.Image.Image]]

    model = model.eval().to(device)
    dtype = next(model.parameters()).dtype

    feats = []

    class_feats = []

    # Calculate the total number of batches
    total_batches = ceil(len(dl) / batch_size)

    with torch.inference_mode():
        for batch in tqdm(batched_dl, leave=False, total=total_batches):
            batch: list[PIL.Image.Image]

            if model_name == "hibou-b" or model_name == "hibou-l" or model_name == "phikon":
                # Ensure the batch is correctly normalized
                hf_data = processor(
                    images=batch,
                    return_tensors="pt"
                ).to(device)

                hf_output = model(**hf_data)
                output = hf_output.pooler_output # Use last_hidden_state for detailed spatial info
                feats.append(output.cpu().detach().half())

            elif model_name == "conch":
                processed_images = []
                for image in batch:
                    processed = processor(image)
                    processed_images.append(processed)
                processed_batch = torch.stack(processed_images).to(device, dtype=dtype)
                output = model.encode_image(processed_batch, proj_contrast=False, normalize=False)
                feats.append(output.cpu().detach().half())

            elif model_name == "virchow" or model_name == "virchow2":
                processed_batch = torch.stack([transform(img) for img in batch]).to(device, dtype=dtype)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(processed_batch.type(dtype).to(device))

                class_token = output[:, 0]
                if model_name == "virchow":
                    patch_tokens = output[:, 1:]
                else:
                    patch_tokens = output[:, 5:]
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                embedding = embedding.to(torch.float16)
                feats.append(embedding)
                class_token = class_token.to(torch.float16)
                class_feats.append(class_token)
            else:
                processed_batch = torch.stack([transform(img) for img in batch]).to(device, dtype=dtype)
                feats.append(
                    model(processed_batch.type(dtype).to(device)).half().cpu().detach())

        all_feats = torch.concat(feats)
        if model_name == "virchow" or model_name == "virchow2":
            all_class_feats = torch.concat(class_feats)

    with h5py.File(f'{outdir}.h5', 'w') as f:
        f['coords'] = coords
        if model_name == "mahmood-uni-56ef09b4":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1024, all_feats.shape
        elif model_name == "xiyuewang-ctranspath-7c998680":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "chief-ctp":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "prov-gigapath":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1536, all_feats.shape
        elif model_name == "hibou-b":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "hibou-l":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1024, all_feats.shape
        elif model_name == "kaiko-vitl14":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1024, all_feats.shape
        elif model_name == "conch":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 512, all_feats.shape
        elif model_name == "phikon":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "virchow" or model_name == "virchow2":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 2560, all_feats.shape
            assert len(all_class_feats.shape) == 2 and all_class_feats.shape[1] == 1280, all_feats.shape
            with h5py.File(f'{outdir}_class_tokens.h5', 'w') as g:
                g['coords'] = coords
                g['feats'] = all_class_feats.cpu().numpy()
                g['augmented'] = np.repeat([False, True], [len(unaugmented_ds), len(augmented_ds)])
                g.attrs['extractor'] = extractor_string
        elif model_name == "hoptimus0":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1536, all_feats.shape
        else:
            print(f"Model name did not match any known patterns: {model_name}")
            print(f"Shape of all_feats: {all_feats.shape}")
            raise ValueError(f"Unknown model name: {model_name}")
        f['feats'] = all_feats.cpu().numpy()
        f['augmented'] = np.repeat(
            [False, True], [len(unaugmented_ds), len(augmented_ds)])
        assert len(f['feats']) == len(f['augmented'])
        f.attrs['extractor'] = extractor_string