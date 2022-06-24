
# 3.2.1 ベルヌーイ分布の学習と予測 ----------------------------------------------------------------

# 3.2.1項で利用するパッケージ
library(tidyverse)


### 尤度(ベルヌーイ分布)の設定 -----

# 真のパラメータを指定
mu_truth <- 0.25


# 尤度を計算:式(2.16)
model_df <- tibble(
  x = c(0, 1), # xがとり得る値
  prob = c(1 - mu_truth, mu_truth) # 確率
  #prob = mu_truth^x * (1 - mu_truth)^(1 - x) # 確率
  #prob = dbinom(x = x, size = 1, prob = mu_truth) # 確率
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "blue") + # 尤度
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = paste0("mu=", mu_truth))


### データの生成 -----

# (観測)データ数を指定
N <- 50

# ベルヌーイ分布に従うデータを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)

# 観測データを確認
table(x_n)


# 観測データのデータフレームを作成
data_df <- tibble(x_n = x_n)

# 観測データのヒストグラムを作成
ggplot() + 
  #geom_histogram(data = data_df, aes(x = x_n), binwidth = 1) + # 観測データ:(度数)
  geom_histogram(data = data_df, aes(x = x_n, y = ..density..), binwidth = 1) + # 観測データ:(相対度数)
  geom_bar(data = model_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Bernoulli Distribution", 
       subtitle = paste0("N=", N, ", mu=", mu_truth), 
       x = "x")


### 事前分布(ベータ分布)の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のmuの値を作成
mu_vec <- seq(0, 1, by = 0.001)

# 事前分布を計算:式(2.41)
prior_df <- tibble(
  mu = mu_vec, # x軸の値
  ln_C_Beta = lgamma(a + b) - lgamma(a) - lgamma(b), # 正規化項(対数)
  density = exp(ln_C_Beta + (a - 1) * log(mu) + (b - 1) * log(1 - mu)) # 確率密度
  #density = dbeta(x = mu, shape1 = a, shape2 = b) # 確率密度
)

# 事前分布を作図
ggplot(prior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # 事前分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Beta Distribution", 
       subtitle = paste0("a=", a, ", b=", b), 
       x = expression(mu))


### 事後分布(ベータ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.15)
a_hat <- sum(x_n) + a
b_hat <- N - sum(x_n) + b


# 事後分布を計算:式(2.41)
posterior_df <- tibble(
  mu = mu_vec, # x軸の値
  ln_C_Beta = lgamma(a_hat + b_hat) - lgamma(a_hat) - lgamma(b_hat), # 正規化項(対数)
  density = exp(ln_C_Beta + (a_hat - 1) * log(mu) + (b_hat - 1) * log(1 - mu)) # 確率密度
  #density = dbeta(x = mu, shape1 = a_hat, shape2 = b_hat) # 確率密度
)

# 事後分布を作図
ggplot(posterior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # 事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Beta Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", b_hat), 
       x = expression(mu))


### 予測分布(ベルヌーイ分布)の計算 -----

# 予測分布のパラメータを計算:式(3.19')
mu_star_hat <- a_hat / (a_hat + b_hat)
#mu_star_hat <- (sum(x_n) + a) / (N + a + b)


# 予測分布を計算:式(2.16)
predict_df <- tibble(
  x = c(0, 1), # xがとり得る値
  prob = c(1 - mu_star_hat, mu_star_hat) # 確率
  #prob = mu_star_hat^x * (1 - mu_star_hat)^(1 - x) # 確率
  #prob = dbinom(x = x, size = 1, prob = mu_star_hat) # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, aes(x =x, y = prob), 
           stat = "identity", position = "dodge", fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Beta Distribution", 
       subtitle = paste0("N=", N, ", mu_star_hat=", round(mu_star_hat, 5)))


# ・アニメーションによる推移の確認 -----------------------------------------------------------------

# 3.2.1項で利用するパッケージ
library(tidyverse)
library(gganimate)


### モデルの設定 -----

# 真のパラメータを指定
mu_truth <- 0.25

# 事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のmuの値を作成
mu_vec <- seq(0, 1, by = 0.001)

# 事前分布(ベータ分布)を計算:式(2.41)
posterior_df <- tibble(
  mu = mu_vec, # x軸の値
  density = dbeta(x = mu, shape1 = a, shape2 = b), # 確率密度
  label = as.factor(paste("N=", 0, ", a=", a, ", b", b)) # フレーム切替用のラベル
)


# 初期値による予測分布のパラメーターを計算:式(3.19)
mu_star <- a / (a + b)


# 初期値による予測分布(ベルヌーイ分布)を計算:式(2.16)
predict_df <- tibble(
  x = c(0, 1), # xがとり得る値
  prob = c(1 - mu_star, mu_star), # 対応する確率
  label = as.factor(paste0("N=", 0, ", mu_star=", round(mu_star, 2))) # フレーム切替用のラベル
)


### 推論処理 -----

# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を作成
x_n <- rep(0, N)

# ベイズ推論
for(n in 1:N){
  
  # ベルヌーイ分布に従うデータを生成
  x_n[n] <- rbinom(n = 1, size = 1, prob = mu_truth)
  
  # 事後分布のパラメータを更新:式(3.15)
  a <- x_n[n] + a
  b <- 1 - x_n[n] + b
  
  # 事後分布(ベータ分布)を計算:式(2.41)
  tmp_posterior_df <- tibble(
    mu = mu_vec, # x軸の値
    density = dbeta(x = mu, shape1 = a, shape2 = b), # 確率密度
    label = as.factor(paste0("N=", n, ", a_hat=", a, ", b_hat=", b)) # フレーム切替用のラベル
  )
  
  # 予測分布のパラメーターを更新:式(3.19)
  mu_star <- a / (a + b)
  
  # 予測分布(ベルヌーイ分布)を計算:式(2.16)
  tmp_predict_df <- tibble(
    x = c(0, 1), # xがとり得る値
    prob = c(1 - mu_star, mu_star), # 確率
    label = as.factor(paste0("N=", n, ", mu_star_hat=", round(mu_star, 2))) # フレーム切替用のラベル
  )
  
  # n回目の結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
table(x_n)


### 作図処理 -----

# 観測データのデータフレームを作成
data_df <- tibble(
  x_n = c(NA, x_n), 
  label = unique(posterior_df[["label"]]) # フレーム切替用のラベル
)

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_line(data = posterior_df, aes(x = mu, y = density), 
            color = "purple") + # 事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真の値
  geom_point(data = data_df, aes(x = x_n, y = 0), size = 5) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Beta Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = N + 1, fps = 10)


# 尤度をN+1フレーム分に複製
model_df <- tibble(
  x = rep(c(0, 1), times = N + 1), # x軸の値
  prob = rep(c(1 - mu_truth, mu_truth), times = N + 1), # 確率
  label = predict_df[["label"]] # フレーム切替用のラベル
)

# 観測データのデータフレームを作成
data_df <- tibble(
  x_n = c(NA, x_n), 
  label = unique(predict_df[["label"]]) # フレーム切替用のラベル
)

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_bar(data = predict_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  geom_point(data = data_df, aes(x = x_n, y = 0), size = 5) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = N + 1, fps = 10)


