# logo_detection

## Environment：

PaddleDetection

PaddleOCR


## Dataset：

[Weblogo_500](https://pan.baidu.com/s/1TCGQfi0bjNBvIKyf9U_ynA?pwd=logo )

[Weblogo_1000](https://pan.baidu.com/s/101mTA7f6bpUvjI4uftTXnA?pwd=logo )

提取码：logo 


## Usage：

* Download the weight file from paddlepaddle. Modify the weight path and dataset path in configs. 

* train: `python -u tools/train.py -c ./configs/logomask/logomask.yml -o weights=your path --eval`

* eval: `python -u tools/eval.py -c -c ./configs/logomask/logomask.yml -o weights=your path  --classwise`

