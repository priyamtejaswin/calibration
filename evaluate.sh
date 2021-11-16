export TRAIN_PATH="output/dev/QQP_QQP_roberta-base.json"
export TEST_PATH="output/test/QQP_QQP_roberta-base.json"

python3 calibrate.py \
	--train_path $TRAIN_PATH \
	--test_path $TEST_PATH \
	--do_train \
	--do_evaluate
