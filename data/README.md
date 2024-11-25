# M5B Image Names to Dollar Street IDs

The two datasets introduced by the M5 Benchmark, i.e., `m5b_vgr` and `m5b_vlod`, comprise images from the Dollar Street dataset (see citation below).

Due to the large resolution of the original images, we downsampled them to 640x480 pixels and assigned new identifiers (in the column `image_names`) to the images.

A mapping from the M5 `image_names` to the original Dollar Street IDs is provided in the `m5b_image_name_to_dollar_street_id.csv` file in this directory.

## Dollar Street Citation

```bibtex
@article{gaviria2022dollar,
  title={The Dollar Street dataset: Images representing the geographic and socioeconomic diversity of the world},
  author={Gaviria Rojas, William and Diamos, Sudnya and Kini, Keertan and Kanter, David and Janapa Reddi, Vijay and Coleman, Cody},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={12979--12990},
  year={2022}
}
```
