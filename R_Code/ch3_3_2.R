
# 3.3.2 1次元ガウス分布の学習と予測：精度が未知の場合 ------------------------------------------------------------------

# 3.3.2項で利用するパッケージ
library(tidyverse)


### 尤度(ガウス分布)の設定 -----

# 真のパラメータを指定
mu <- 25
lambda_truth <- 0.01

# 作図用のxの値を設定
x_line <- seq(
  mu - 4 * sqrt(1 / lambda_truth), 
  mu + 4 * sqrt(1 / lambda_truth), 
  by = 0.1
)

# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_line, # x軸の値
  density = dnorm(x, mean = mu, sd = sqrt(1 / lambda_truth)) # 確率密度
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = density)) + 
  geom_line(color = "purple") + # 尤度
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("mu=", round(mu, 2), ", sigma=", round(sqrt(1 / lambda_truth), 1)))


### 観測データの生成 -----

# (観測)データ数を指定
N <- 50

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu, sd = sqrt(1 / lambda_truth))

# 観測データを確認
summary(x_n)

# 観測データのヒストグラムを作図
tibble(x = x_n) %>% 
  ggplot(aes(x = x)) + 
  geom_histogram(binwidth = 1) + # 観測データ
  labs(title = "Observation Data", 
       subtitle = paste0("N=", N, ", mu=", mu, ", sigma=", round(sqrt(1 / lambda_truth), 1)))


### 事前分布(ガンマ分布)の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1

# 作図用のlambdaの値を設定
lambda_line <- seq(0, 4 * lambda_truth, by = 0.00001)

# 事後分布を計算:式(2.56)
prior_df <- tibble(
  lambda = lambda_line, # x軸の値
  density = dgamma(x = lambda, shape = a, scale = 1 / b) # 確率密度
)

# 事前分布を作図
ggplot(prior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # 事前分布
  labs(title = "Gamma Distribution", 
       subtitle = paste0("a=", a, ", b=", b))


### 事後分布(ガウス分布)の計算 -----

# 事後分布のパラメータを計算:式(3.69)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * sum((x_n - mu)^2) + b

# 事後分布を計算:式(2.56)
posterior_df <- tibble(
  lambda = lambda_line, # x軸の値
  ln_C_gam = a_hat * log(b_hat) - lgamma(a_hat), # 正規化項(対数)
  #density = exp(ln_C_gam + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
  density = dgamma(x = lambda, shape = a_hat, scale = 1 / b_hat) # 確率密度
)

# 事後分布を作図
ggplot(posterior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # 事前分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真のパラメータ
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", round(b_hat, 1)), 
       x = expression(lambda))


### 予測分布(スチューデントのt分布)の計算 -----

# 予測分布のパラメータを計算:式(3.79)
mu_s <- mu
lambda_s_hat <- a_hat / b_hat
nu_s_hat <- 2 * a_hat
#lambda_s_hat <- (N + 2 * a) / (sum((x_n - mu)^2) + 2 * b)
#nu_s_hat <- N + 2 * a

# 予測分布を計算:式(3.76)
predict_df <- tibble(
  x = x_line, # x軸の値
  C_St = lgamma((nu_s_hat + 1) / 2) - lgamma(nu_s_hat / 2),  # 正規化項(対数)
  term1 = log(lambda_s_hat / pi / nu_s_hat) / 2, 
  term2 = - (nu_s_hat + 1) / 2 * log(1 + lambda_s_hat / nu_s_hat * (x - mu_s)^2), 
  density = exp(C_St + term1 + term2) # 確率密度
)

# 予測分布を作図
ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = density), 
            color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Student's t Distribution", 
       subtitle = paste0("N=", N, ", mu_s=", mu_s, 
                         ", lambda_s_hat=", round(lambda_s_hat, 3), 
                         ", nu_s_hat=", nu_s_hat))


# 3.3.2 try ---------------------------------------------------------------

predict_df <- tibble(
  x = seq(0, 2 * mu_s, by = 0.01),  # 作図用の値
  t = x - mu_s,  # 中心をズラす
  density = dt(t, df = nu_s_hat) # 確率密度
)


# ・アニメーション ---------------------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(gganimate)


### 推論処理 -----

# 真のパラメータを指定
mu <- 25
lambda_truth <- 0.01

# 事前分布のパラメータを指定
a <- 1
b <- 1

# 作図用のlambdaの値を設定
lambda_line <- seq(0, 4 * lambda_truth, by = 0.00002)

# 事前分布を計算
posterior_df <- tibble(
  lambda = lambda_line, # x軸の値
  C_G = a * log(b) - lgamma(a),  # 正規化項(対数)
  density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
  label = as.factor(paste0("N=", 0, ", a=", a, ", b=", round(b, 1))) # パラメータ
)

# 初期値による予測分布のパラメータを計算:式(3.79)
mu_s <- mu
lambda_s <- a / b
nu_s <- 2 * a

# 作図用のxの値を設定
x_line <- seq(
  mu - 4 * sqrt(1 / lambda_truth), 
  mu + 4 * sqrt(1 / lambda_truth), 
  by = 0.1
)

# 初期値による予測分布を計算:式(3.76)
predict_df <- tibble(
  x = seq(-mu_s, 3 * mu_s, by = 0.01),  # 作図用の値
  C_St = lgamma((nu_s + 1) / 2) - lgamma(nu_s / 2),  # 正規化項(対数)
  term1 = log(lambda_s / pi / nu_s) / 2, 
  term2 = - (nu_s + 1) / 2 * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
  density = exp(C_St + term1 + term2), # 確率密度
  label = as.factor(
    paste0("N=", 0, ", mu_s=", mu_s, ", lambda_s=", round(lambda_s, 3), ", nu_s", nu_s)
  ) # パラメータ
)

# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を初期化
x_n <- rep(0, N)

# ベイズ推論
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu, sd = sqrt(1 / lambda_truth))
  
  # 事後分布のパラメータを更新:式(3.69)
  a <- 1 / 2 + a
  b <- (x_n[n] - mu)^2 / 2 + b
  
  # 事後分布を計算:式(2.56)
  tmp_posterior_df <- tibble(
    lambda = lambda_line, # x軸の値
    C_G = a * log(b) - lgamma(a),  # 正規化項(対数)
    density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
    label = as.factor(paste0("N=", n, ", a_hat=", a, ", b_hat=", round(b, 1))) # パラメータ
  )
  
  # 予測分布のパラメータを更新:式(3.79)
  mu_s <- mu
  lambda_s <- a / b
  nu_s <- 2 * a
  
  # 予測分布を計算:式(3.76)
  tmp_predict_df <- tibble(
    x = x_line, # x軸の値
    C_St = lgamma((nu_s + 1) / 2) - lgamma(nu_s / 2),  # 正規化項(対数)
    term1 = log(lambda_s / pi / nu_s) / 2, 
    term2 = - (nu_s + 1) / 2 * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
    density = exp(C_St + term1 + term2), # 確率密度
    label = as.factor(
      paste0("N=", n, ", mu_s_hat=", mu_s, 3, ", lambda_s_hat=", round(lambda_s, 3), ", nu_s_hat=", nu_s)
    ) # パラメータ
  )
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### 作図処理 -----

# 事後分布を作図
posterior_graph <- ggplot(posterior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # 事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真のパラメータ
  transition_manual(label) + # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda))

# gif画像を出力
animate(posterior_graph, nframes = N + 1, fps = 10)


# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_line, # x軸の値
  density = dnorm(x, mean = mu, sd = sqrt(1 / lambda_truth)) # 確率密度
)

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = density), 
            color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  transition_manual(label) + # フレーム
  labs(title = "Student's t Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
animate(predict_graph, nframes = N + 1, fps = 10)


