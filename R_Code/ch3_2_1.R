
# 3.2.1 ベルヌーイ分布の学習と予測 ----------------------------------------------------------------

# 3.2.1項で利用するパッケージ
library(tidyverse)


### 真のモデルの設定 -----

# 真のパラメータを指定
mu_truth <- 0.25

# 真のモデル(ベルヌーイ分布)を計算
model_df <- tibble(
  x = c(0, 1), # xが取り得る値
  prob = c(1 - mu_truth, mu_truth) # 対応する確率
  #prob = mu_truth^x * (1 - mu_truth)^(1 - x) # 確率
  #prob = dbinom(x = x, size = 1, prob = mu_truth) # 確率
)

# 真のモデルを作図
ggplot(model_df, aes(x = x, y = prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "purple") + # 棒グラフ
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = paste0("mu=", mu_truth))


### 観測データの生成 -----

# データ数を指定
N <- 50

# (観測)データを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)

# 観測データを確認
table(x_n)

# 観測データのヒストグラムを作図
tibble(x = x_n) %>% 
  ggplot(aes(x = x)) + 
    geom_histogram(bins = 2) + # ヒストグラム
    labs(title = "Observation Data", 
         subtitle = paste0("N=", N, ", mu=", mu_truth))


### 事前分布の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1

# 事前分布(ベータ分布)を計算
prior_df <- tibble(
  mu = seq(0, 1, by = 0.001), # muが取り得る値
  ln_C_beta = lgamma(a + b) - lgamma(a) - lgamma(b), # 正規化項(対数)
  density = exp(ln_C_beta) * mu^(a - 1) * (1 - mu)^(b - 1) # 確率密度
  #density = dbeta(x = mu, shape1 = a, shape2 = b) # 確率密度
)

# 事前分布を作図
ggplot(prior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # 折れ線グラフ
  labs(title = "Prior Distribution", 
       subtitle = paste0("N=", N, ", a=", a, ", b=", b), 
       x = expression(mu))


### 事後分布の計算 -----

# 事後分布のパラメータを計算
a_hat <- sum(x_n) + a
b_hat <- N - sum(x_n) + b

# 事後分布(ベータ分布)を計算
posterior_df <- tibble(
  mu = seq(0, 1, by = 0.001), # 作図用の値
  ln_C_beta = lgamma(a_hat + b_hat) - lgamma(a_hat) - lgamma(b_hat), # 正規化項(対数)
  density = exp(ln_C_beta) * mu^(a_hat - 1) * (1 - mu)^(b_hat - 1) # 確率密度
  #density = dbeta(x = mu, shape1 = a_hat, shape2 = b_hat) # 確率密度
)

# 事後分布を作図
ggplot(posterior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # 事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真のパラメータ
  labs(title = "Beta Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", b_hat), 
       x = expression(mu))


### 予測分布 -----

# 予測分布のパラメーターを計算
mu_hat_star <- a_hat / (a_hat + b_hat)
#mu_hat_star <- (sum(x_n) + a) / (N + a + b)

# 予測分布(ベルヌーイ分布)を計算
predict_df <- tibble(
  x = c(0, 1), # xが取り得る値
  prob = c(1 - mu_hat_star, mu_hat_star) # 対応する確率
  #prob = mu_hat_star^x * (1 - mu_hat_star)^(1 - x) # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, aes(x =x, y = prob), stat = "identity", position = "dodge", 
           fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真のモデル
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Predictive Distribution", 
       subtitle = paste0("N=", N, ", mu_hat_star=", round(mu_hat_star, 2)))


# アニメーション -----------------------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(gganimate)


### 推論処理 -----

# 真のパラメータを指定
mu_truth <- 0.25

# 事前分布のパラメータを指定
a <- 1
b <- 1

# 事前分布(ベータ分布)を計算
posterior_df <- tibble(
  mu = seq(0, 1, by = 0.001), # muが取り得る値
  density = dbeta(x = mu, shape1 = a, shape2 = b), # 確率密度
  label = as.factor(paste("N=", 0, ", a=", a, ", b", b)) # 試行回数とパラメータのラベル
)

# 初期値による予測分布のパラメーターを計算
mu_star <- a / (a + b)

# 初期値による予測分布(ベルヌーイ分布)を計算
predict_df <- tibble(
  x = c(0, 1), # xが取り得る値
  prob = c(1 - mu_star, mu_star), # 対応する確率
  label = as.factor(paste0("N=", 0, ", mu_hat_star=", round(mu_star, 2))) # 試行回数とパラメータのラベル
)


# データ数(試行回数)を指定
N <- 100

# 推論処理
x_n <- rep(0, N) # 受け皿を初期化
for(n in 1:N){
  
  # ベルヌーイ分布に従うデータを生成
  x_n[n] <- rbinom(n = 1, size = 1, prob = mu_truth)
  
  # 事後分布のパラメータを更新
  a <- x_n[n] + a
  b <- 1 - x_n[n] + b
  
  # 事後分布を計算
  tmp_posterior_df <- tibble(
    mu = seq(0, 1, by = 0.001), # muがとり得る値
    density = dbeta(x = mu, shape1 = a, shape2 = b), # 確率密度
    label = as.factor(paste0("N=", n, ", a_hat=", a, ", b_hat=", b)) # 試行回数とパラメータのラベル
  )
  
  # 予測分布のパラメーターを更新
  mu_star <- a / (a + b)
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = c(0, 1),  # xが取り得る値
    prob = c(1 - mu_star, mu_star), # 対応する確率
    label = as.factor(paste0("N=", n, ", mu_hat_star=", round(mu_star, 2))) # 試行回数とパラメータのラベル
  )
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
table(x_n)


### 作図 -----

# 事後分布を作図
posterior_graph <- ggplot(posterior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") +  # 事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真のパラメータ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Beta Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = (N + 1), fps = 10)


# Nフレーム分の真のモデルを格納したデータフレームを作成
label_vec <- unique(predict_df[["label"]])
model_df <- tibble()
for(n in 1:(N + 1)) {
  # n番目のフレーム用に計算
  tmp_df <- tibble(
    x = c(0, 1), 
    prob = c(1 - mu_truth, mu_truth), 
    label = label_vec[n]
  )
  
  # 結果を結合
  model_df <- rbind(model_df, tmp_df)
}

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_bar(data = predict_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = x, y = prob), stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真のモデル
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Bernoulli Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = (N + 1), fps = 10)


