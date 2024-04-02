
# gen single object subset
python generate_pvd_img_svg.py \
    --dataset "single_obj" \
    --split "train" \

# gen outlined compositional objects subset
python generate_pvd_img_svg.py \
    --dataset "multi_obj" \
    --split "train" \
    --multi_obj_style "outline"

# gen filled compositional objects subset
python generate_pvd_img_svg.py \
    --dataset "multi_obj" \
    --split "train" \
    --multi_obj_style "filled"