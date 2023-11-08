from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import hiera
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


model = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")  # Checkpoint is optional (will use default)


# Create input transformations
input_size = 224

transform_list = [
    transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(input_size)
]

# The visualization and model need different transforms
transform_vis  = transforms.Compose(transform_list)
transform_norm = transforms.Compose(transform_list + [
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


# Load the image
img = Image.open("./img/dog.jpg")
img_vis = transform_vis(img)
img_norm = transform_norm(img)

# Get imagenet class as output
out = model(img_norm[None, ...]) # add batch dim

# 207: golden retriever  (imagenet-1k)
out.argmax(dim=-1).item()


# If you also want intermediate feature maps
_, intermediates = model(img_norm[None, ...], return_intermediates=True)

for x in intermediates:
    print(x.shape)