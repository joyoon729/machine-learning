# Introduction To Ensemble Learning - Boosting Algorithm



## 0. Intro

머신러닝 **앙상블(Ensemble) 학습**의  **Boosting Algorithm** 중 대표적인 알고리즘 몇가지를 알아봅니다.

- AdaBoost
- Gradient Boost Machine (GBM)
- XGBoost
- LightGBM
- CatBoost



## 1. AdaBoost

Adaptive Boosting 의 줄임말인 **AdaBoost** 알고리즘은 **Boosting** 알고리즘의 가장 기본적인 알고리즘입니다.

**AdaBoost** 알고리즘은 다음의 단계로 진행됩니다.

1. 각 weak 모델에서 학습합니다.
2. 학습 후 오류(error)를 계산합니다.
3. 이 오류를 통해 가장 작은 오류를 낸 모델에 높은 가중치를 줍니다.
4. 1~3 과정을 반복하며 