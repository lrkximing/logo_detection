# logo_detection

##Environment：
PaddleDetection
PaddleOCR

##Dataset：
Weblogo_500:
Weblogo_1000:


##Usage：
1.Download the weight file from paddlepaddle. Modify the weight path and dataset path in configs. 
2.train: python -u tools/train.py -c ./configs/logomask/logomask.yml -o weights=your path --eval
3.eval: python -u tools/eval.py -c -c ./configs/logomask/logomask.yml -o weights=your path  --classwise

