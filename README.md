# 概述
> 采用差分学习率，


# 运行
- 数据格式见：/data/长文本分类

### 运行
- python train.py 
在colab上跑了两个epoch之后的结果为：
test result is 0.94
              precision    recall  f1-score   support

          教育       0.96      0.96      0.96        56
          家居       0.92      0.88      0.90        50
          时尚       1.00      0.98      0.99        50
          时政       0.91      0.96      0.94        53
          科技       0.95      0.95      0.95        41
          房产       0.91      0.94      0.93        54
          财经       0.93      0.89      0.91        46

    accuracy                           0.94       350
   macro avg       0.94      0.94      0.94       350
weighted avg       0.94      0.94      0.94       350

# 使用差分学习率
- python train.py --optim=True
test result is 0.9342857142857143
              precision    recall  f1-score   support

          教育       0.95      0.96      0.96        56
          家居       0.93      0.84      0.88        50
          时尚       1.00      0.98      0.99        50
          时政       0.94      0.94      0.94        53
          科技       0.93      0.95      0.94        41
          房产       0.86      0.93      0.89        54
          财经       0.93      0.93      0.93        46

    accuracy                           0.93       350
   macro avg       0.94      0.93      0.93       350
weighted avg       0.94      0.93      0.93

