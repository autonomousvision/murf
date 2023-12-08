# Model Zoo

- The models are hosted on Hugging Face 🤗 : https://huggingface.co/haofeixu/murf
- `RealEstate10K (256x256)` is the 256x256 resolution RealEstate10K dataset following [AttnRend](https://github.com/yilundu/cross_attention_renderer) for fair comparison.
- `MIX1` denotes the mixed datasets of `ibrnet_collected`, `LLFF(training set)`, `Spaces`, `RealEstate10K (200-scene subset)` and `Google Scanned Objects` following [IBRNet](https://github.com/googleinterns/IBRNet#1-training-datasets) for fair comparison. The resolution of the `RealEstate10K` dataset here is 720x1280.

- `MIX2` denotes the mixed datasets of `ibrnet_collected`, `LLFF (training set)`, `Spaces` and `RealEstate10K (10000-scene subset)` . The key difference with `MIX1` is the use of larger RealEstate10K subset (10000 vs. 200). 
- The `MuRF-mixdata` model is recommended for in-the-wild use cases.

| Model                              |         Training Data          | Training Views |                           Download                           |
| ---------------------------------- | :----------------------------: | :------------: | :----------------------------------------------------------: |
| MuRF-dtu-small-baseline-2view      |              DTU               |       2        | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-dtu-small-baseline-2view-21d62708.pth) |
| MuRF-dtu-small-baseline-3view      |              DTU               |       3        | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-dtu-small-baseline-3view-ecc90367.pth) |
| MuRF-dtu-large-baseline-6view      |              DTU               |       6        | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-dtu-large-baseline-6view-c52d3b16.pth) |
| MuRF-dtu-large-baseline-9view      |              DTU               |       9        | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-dtu-large-baseline-9view-6754a597.pth) |
| MuRF-realestate10k-2view           |    RealEstate10K (256x256)     |       2        | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-realestate10k-2view-74b3217d.pth) |
| MuRF-llff-6view                    |              MIX1              |       6        | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-llff-6view-15d3646e.pth) |
| MuRF-llff-10view                   |              MIX1              |       10       | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-llff-10view-d74cff18.pth) |
| MuRF-mipnerf360-2view-42df3b73.pth | RealEstate10K (256x256) & MIX1 |       2        | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-mipnerf360-2view-42df3b73.pth) |
| MuRF-mixdata                       | RealEstate10K (256x256) & MIX2 |   random 2~6   | [download](https://huggingface.co/haofeixu/murf/resolve/main/murf-mixdata-51859ce2.pth) |


