global="--group=max_min_add_sub_mul_scaleadd_matrixmul_cyclicadd --max_epochs=10 --max_literal_n=10"

for i in $(seq 0 28)
do
    python main/train.py ${global} --algebra=transported --run_name=transported${i} --transported_index=${i}
done

python main/train.py ${global} --algebra=directparam --run_name=directparam --directparam_algebra_symmetric=False
python main/train.py ${global} --algebra=directparam --run_name=directparam_symmetric --directparam_algebra_symmetric=True
