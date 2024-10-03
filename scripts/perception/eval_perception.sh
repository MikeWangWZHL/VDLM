MODEL_DIR=data/ckpts
DATA_ROOT=data/datasets/downstream_tasks

### Assuming the server is setup
OUTPUT_ROOT=results/perception


# low-level reasoning tasks
TASKS="acute-or-obtuse length-comparison nlvr shapeworld-spatial-2obj shapeworld-spatial-multiobj shapeworld-superlative geoclidean-2shot maze-solve-2x2 maze_solve-3x3"

# # high-level reasoning tasks
# TASKS="vgbench_qa_svg_category vgbench_qa_svg_color vgbench_qa_svg_usage"


for TASK in $TASKS
do
    echo "TASK=$TASK"
    INPUT_JSON=${DATA_ROOT}/${TASK}.json
    OUTPUT_DIR=${OUTPUT_ROOT}/${TASK}
    K=-1
    python scripts/perception/eval_perception.py \
        --data_root $DATA_ROOT \
        --input_json $INPUT_JSON \
        --output_dir $OUTPUT_DIR \
        --k $K
done