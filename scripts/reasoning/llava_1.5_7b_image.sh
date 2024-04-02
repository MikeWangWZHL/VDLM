TEMPERATURE=0.0

CUDA_ID=0
MODEL_PATH="liuhaotian/llava-v1.5-7b"
MODEL_NAME="llava-v1.5-7b"

export DATA_ROOT="data/datasets/downstream_tasks"
DATASET_NAME="None"
VERSION="None"

TASKS="acute-or-obtuse length-comparison nlvr shapeworld-spatial-2obj shapeworld-spatial-multiobj shapeworld-superlative geoclidean-2shot maze-solve-2x2 maze_solve-3x3"

INPUT_TYPE="image"
for TASK in $TASKS
do
    TASK_NAME="${TASK}__${INPUT_TYPE}"
    OUTPUT_DIR="./results/reasoning/${TASK_NAME}__${MODEL_NAME}"
    echo "Task name: ${TASK_NAME}"
    CUDA_VISIBLE_DEVICES=${CUDA_ID} python llava_inference.py \
        --model-path ${MODEL_PATH} \
        --model-name ${MODEL_NAME} \
        --temperature ${TEMPERATURE} \
        --output-dir ${OUTPUT_DIR} \
        --run-task ${TASK_NAME}  \
        --dataset-name ${DATASET_NAME} \
        --version ${VERSION}
done

