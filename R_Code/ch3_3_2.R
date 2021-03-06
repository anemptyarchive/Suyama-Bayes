
# 3.3.2 1次元ガウス分布の学習と予測：精度が未知の場合 ------------------------------------------------------------------

# 3.3.2項で利用するパッケージ
library(tidyverse)


### 尤度(ガウス分布)の設定 -----

# 真のパラメータを指定
mu <- 25
lambda_truth <- 0.01
print(sqrt(1 / lambda_truth)) # 標準偏差

# 作図用のxの値を設定
x_line <- seq(
  mu - 4 * sqrt(1 / lambda_truth), 
  mu + 4 * sqrt(1 / lambda_truth), 
  length.out = 1000
)

# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_line, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(lambda_truth)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * lambda_truth * (x - mu)^2) # 確率密度
  #C_N = 1 / sqrt(2 * pi / lambda_truth), # 正規化項
  #density = C_N * exp(- 0.5 * lambda_truth * (x - mu)^2) # 確率密度
  #density = dnorm(x, mean = mu, sd = sqrt(1 / lambda_truth)) # 確率密度
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = density)) + 
  geom_line(color = "purple") + # 尤度
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("mu=", round(mu, 2), ", lambda=",lambda_truth))


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
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu=", mu, ", sigma=", round(sqrt(1 / lambda_truth), 1)))


### 事前分布(ガンマ分布)の設定 -----

# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1

# 作図用のlambdaの値を設定
lambda_line <- seq(0, 4 * lambda_truth, length.out = 1000)

# lambdaの事前分布を計算:式(2.56)
prior_df <- tibble(
  lambda = lambda_line, # x軸の値
  ln_C_Gam = a * log(b) - lgamma(a), # 正規化項(対数)
  density = exp(ln_C_Gam + (a - 1) * log(lambda) - b * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)

# lambdaの事前分布を作図
ggplot(prior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事前分布
  labs(title = "Gamma Distribution", 
       subtitle = paste0("a=", a, ", b=", b))


### 事後分布(ガンマ分布)の計算 -----

# lambdaの事後分布のパラメータを計算:式(3.69)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * sum((x_n - mu)^2) + b

# lambdaの事後分布を計算:式(2.56)
posterior_df <- tibble(
  lambda = lambda_line, # x軸の値
  ln_C_Gam = a_hat * log(b_hat) - lgamma(a_hat), # 正規化項(対数)
  density = exp(ln_C_Gam + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a_hat, rate = b_hat) # 確率密度
)

# lambdaの事後分布を作図
ggplot(posterior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真のlambda
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
  ln_C_St = lgamma(0.5 * (nu_s_hat + 1)) - lgamma(0.5 * nu_s_hat), # 正規化項(対数)
  ln_term1 = 0.5 * log(lambda_s_hat / pi / nu_s_hat), 
  ln_term2 = - 0.5 * (nu_s_hat + 1) * log(1 + lambda_s_hat / nu_s_hat * (x - mu_s)^2), 
  density = exp(ln_C_St + ln_term1 + ln_term2) # 確率密度
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


# try ---------------------------------------------------------------

# 予測分布を作図
tibble(
  x = x_line, # x軸の値
  t = x - mu_s, # 中心をズラす
  density = dt(t, df = nu_s_hat) # 確率密度
) %>% 
  ggplot(data = ., aes(x = t, y = density)) + 
  geom_line(color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Student's t Distribution", 
       subtitle = paste0("N=", N, ", mu_s=", mu_s, 
                         ", lambda_s_hat=", round(lambda_s_hat, 3), 
                         ", nu_s_hat=", nu_s_hat))

warnings()

# ・アニメーション ---------------------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(gganimate)


### 推論処理 -----

# 真のパラメータを指定
mu <- 25
lambda_truth <- 0.01

# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のlambdaの値を設定
lambda_line <- seq(0, 4 * lambda_truth, length.out = 1000)

# lambdaの事前分布(ガンマ分布)を計算:式(2.56)
posterior_df <- tibble(
  lambda = lambda_line, # x軸の値
  density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
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
  length.out = 1000
)

# 初期値による予測分布(スチューデントのt分布)を計算:式(3.76)
predict_df <- tibble(
  x = x_line, # x軸の値
  ln_C_St = lgamma(0.5 * (nu_s + 1)) - lgamma(0.5 * nu_s), # 正規化項(対数)
  ln_term1 = 0.5 * log(lambda_s / pi / nu_s), 
  ln_term2 = - 0.5 * (nu_s + 1) * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
  density = exp(ln_C_St + ln_term1 + ln_term2), # 確率密度
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
  
  # lambdaの事後分布のパラメータを更新:式(3.69)
  a <- 1 / 2 + a
  b <- 0.5 * (x_n[n] - mu)^2 + b
  
  # lambdaの事後分布(ガンマ分布)を計算:式(2.56)
  tmp_posterior_df <- tibble(
    lambda = lambda_line, # x軸の値
    density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    label = as.factor(paste0("N=", n, ", a_hat=", a, ", b_hat=", round(b, 1))) # パラメータ
  )
  
  # 予測分布のパラメータを更新:式(3.79)
  mu_s <- mu
  lambda_s <- a / b
  nu_s <- 2 * a
  
  # 予測分布(スチューデントのt分布)を計算:式(3.76)
  tmp_predict_df <- tibble(
    x = x_line, # x軸の値
    ln_C_St = lgamma(0.5 * (nu_s + 1)) - lgamma(0.5 * nu_s), # 正規化項(対数)
    ln_term1 = 0.5 * log(lambda_s / pi / nu_s), 
    ln_term2 = - 0.5 * (nu_s + 1) * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
    density = exp(ln_C_St + ln_term1 + ln_term2), # 確率密度
    label = as.factor(
      paste0("N=", n, ", mu_s=", mu_s, ", lambda_s_hat=", round(lambda_s, 3), ", nu_s_hat=", nu_s)
    ) # パラメータ
  )
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### 作図処理 -----

# lambdaの事後分布を作図
posterior_graph <- ggplot(posterior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真のlambda
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = N + 1, fps = 10)


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
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Student's t Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = N + 1, fps = 10)


