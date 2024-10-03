MODEL_TYPE="gpt4o"
DATASET_NAME="None"
SUBSET_SIZE=100
SLEEP_RATE=15

export DATA_ROOT="data/datasets/downstream_tasks"
export OPENAI_API_KEY="your api key here"
echo "API key: $OPENAI_API_KEY"

# low-level reasoning tasks
TASKS="acute-or-obtuse length-comparison nlvr shapeworld-spatial-2obj shapeworld-spatial-multiobj shapeworld-superlative geoclidean-2shot maze-solve-2x2 maze_solve-3x3"

# # high-level reasoning tasks
# TASKS="vgbench_qa_svg_category vgbench_qa_svg_color vgbench_qa_svg_usage"

INPUT_TYPE="image_pvd"
for TASK in $TASKS
do
    TASK_NAME="${TASK}__${INPUT_TYPE}"
    OUTPUT_DIR="./results/reasoning/${TASK_NAME}__${MODEL_TYPE}"
    echo "Task name: ${TASK_NAME}"
    python gpt4_v_inference.py \
        --subset_size ${SUBSET_SIZE} \
        --model_type ${MODEL_TYPE} \
        --output-dir ${OUTPUT_DIR} \
        --run-task ${TASK_NAME}  \
        --dataset-name ${DATASET_NAME} \
        --sleep_rate $SLEEP_RATE \
        --max_tokens 4096
done