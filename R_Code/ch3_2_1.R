
# 3.2.1 ベルヌーイ分布の学習と予測 ----------------------------------------------------------------

# 3.2.1 事後分布 ----------------------------------------------------------------


# 利用パッケージ
library(tidyverse)


## パラメータの初期値を指定
# 観測モデルのパラメータ
mu_truth <- 0.25

# 事前分布のパラメータ
a <- 1
b <- 1

# 試行回数
N <- 50


# ベルヌーイ分布に従うデータを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)

# 観測データを確認
table(x_n)


# 事後分布のパラメータを計算
a_hat <- sum(x_n) + a
b_hat <- N - sum(x_n) + b

# 事後分布を計算
posterior_df <- tibble(
  mu = seq(0, 1, by = 0.01),  # 作図用の値
  C_B = lgamma(a_hat + b_hat) - lgamma(a_hat) - lgamma(b_hat),  # 正規化項(対数)
  density = exp(C_B + (a_hat - 1) * log(mu) + (b_hat - 1) * log(1 - mu)) # 確率密度
)


# 作図
ggplot(posterior_df, aes(mu, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  labs(title = "Beta Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", b_hat), 
       x = expression(mu)) # ラベル


# 3.2.1 予測分布 --------------------------------------------------------------


# 予測分布のパラメーターを計算
mu_hat <- a_hat / (a_hat + b_hat)
#mu_hat <- (sum(x_n) + a) / (N + a + b)


# 予測分布を計算
predict_df <- tibble(
  x = c(0, 1),  # 作図用の値
  prob = mu_hat^x * (1 - mu_hat)^(1 - x) # 確率
)


# 作図
ggplot(predict_df, aes(x, prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "#56256E") + # 棒グラフ
  labs(title = "Bernoulli Distribution", 
       subtitle = paste0("N=", N, ", mu_hat=", round(mu_hat, 2))) # ラベル


# 3.2.1 gif ---------------------------------------------------------------------


# 利用パッケージ
library(tidyverse)
library(gganimate)


## パラメータの初期値を指定
# 観測モデルのパラメータ
mu_truth <- 0.25

# 事前分布のパラメータ
a <- 1
b <- 1

# 試行回数
N <- 50


# 事前分布を計算
posterior_df <- tibble(
  mu = seq(0, 1, by = 0.001),  # 作図用の値
  C_B = lgamma(a + b) - lgamma(a) - lgamma(b),  # 正規化項(対数)
  density = exp(C_B + (a - 1) * log(mu) + (b - 1) * log(1 - mu)), # 確率密度
  N = 0  # 試行回数
)


# 初期値による予測分布のパラメーターを計算
mu_star <- a / (a + b)

# 初期値による予測分布を計算
predict_df <- tibble(
  x = c(0, 1),  # 作図用の値
  prob = mu_star^x * (1 - mu_star)^(1 - x),  # 確率
  N = 0 # 試行回数
)


# パラメータを推定
x_n <- rep(0, N) # 受け皿
for(n in 1:N){
  
  # ベルヌーイ分布に従うデータを生成
  x_n[n] <- rbinom(n = 1, size = 1, prob = mu_truth)
  
  
  # ハイパーパラメータを更新
  a <- x_n[n] + a
  b <- 1 - x_n[n] + b
  
  # 事後分布を計算
  tmp_posterior_df <- tibble(
    mu = seq(0, 1, by = 0.001),  # 作図用のmu
    C_B = lgamma(a + b) - lgamma(a) - lgamma(b),  # 正規化項(対数)
    density = exp(C_B + (a - 1) * log(mu) + (b - 1) * log(1 - mu)), # 確率密度
    N = n  # 試行回数
  )
  
  
  # 予測分布のパラメーターを更新
  mu_star <- a / (a + b)
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = c(0, 1),  # 作図用の値
    prob = mu_star^x * (1 - mu_star)^(1 - x),  # 確率
    N = n # 試行回数
  )
  
  
  # 推定結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
table(x_n)


## 事後分布
# 作図
posterior_graph <- ggplot(posterior_df, aes(mu, density)) + 
  geom_line(color = "#56256E") +  # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  transition_manual(N) + # フレーム
  labs(title = "Beta Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(mu)) # ラベル

# 描画
animate(posterior_graph)


## 予測分布
# 作図
predict_graph <- ggplot(predict_df, aes(x, prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "#56256E") + # 棒グラフ
  transition_manual(N) + # フレーム
  labs(title = "Bernoulli Distribution", 
       subtitle = "N= {current_frame}") # ラベル

# 描画
animate(predict_graph)


