cd third_party/viper
DATASET_NAME="None"
SUBSET_SIZE=100
SLEEP_RATE=15

export DATA_ROOT="data/datasets/downstream_tasks"
export OPENAI_API_KEY="your api key here"
echo "API key: $OPENAI_API_KEY"
export CUDA_VISIBLE_DEVICES=0

TASKS="acute-or-obtuse length-comparison nlvr shapeworld-spatial-2obj shapeworld-spatial-multiobj shapeworld-superlative geoclidean-2shot maze-solve-2x2 maze_solve-3x3"

INPUT_TYPE="image"
MODEL_TYPE="vipergpt-gpt4"
for TASK in $TASKS
do
    TASK_NAME="${TASK}__${INPUT_TYPE}"
    OUTPUT_DIR="../../results/reasoning/${TASK_NAME}__${MODEL_TYPE}"
    echo "Task name: ${TASK_NAME}"
    python vipergpt_inference.py \
        --subset_size ${SUBSET_SIZE} \
        --output-dir ${OUTPUT_DIR} \
        --run-task ${TASK_NAME}  \
        --dataset-name ${DATASET_NAME} \
        --sleep_rate $SLEEP_RATE \
        --max_tokens 2048
done