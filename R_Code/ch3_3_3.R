
# 3.3.3 1次元ガウス分布の平均と精度の推定 --------------------------------------------------------------

# 3.3.3で利用するパッケージ
library(tidyverse)


### 尤度(ガウス分布)の設定 -----

# 真のパラメータを指定
mu_truth <- 25
lambda_truth <- 0.01

# 作図用のxの値を設定
x_line <- seq(
  mu_truth - 4 * sqrt(1 / lambda_truth), 
  mu_truth + 4 * sqrt(1 / lambda_truth), 
  by = 0.1
)

# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_line, # x軸の値
  density = dnorm(x = x, mean = mu_truth, sd = sqrt(1 / lambda_truth)) # 確率密度
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = density)) + 
  geom_line(color = "purple") + # 尤度
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("mu=", mu_truth, ", sigma=", round(sqrt(1 / lambda_truth), 1)))


### 観測データの生成 -----

# (観測)データ数を指定
N <- 50

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = sqrt(1 / lambda_truth))

# 観測データを確認
summary(x_n)

# 観測データのヒストグラムを作図
tibble(x = x_n) %>% 
  ggplot(aes(x = x)) + 
    geom_histogram(binwidth = 1) + # 観測データ
    labs(title = "Observation Data", 
         subtitle = paste0("N=", N, ", mu=", mu_truth, ", sigma=", round(sqrt(1 / lambda_truth), 1)))


### 事前分布(ガウス・ガンマ分布)の設定 -----

# muの事前分布のパラメータ
m <- 0
beta <- 1

# lambdaの事前分布のパラメータ
a <- 1
b <- 1

# lambdaの期待値を計算:式(2.59)
E_lambda <- a / b


# 作図用のmuの値を設定
mu_line <- seq(mu_truth - 30, mu_truth + 30, by = 0.01)

# muの事前分布を計算:式(2.64)
prior_mu_df <- tibble(
  mu = mu_line, # x軸の値
  density = dnorm(
    x = mu, mean = m, sd = sqrt(1 / beta / E_lambda)
  ) # 確率密度
)

# muの事前分布を作図
ggplot(prior_mu_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # 事前分布
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("m=", m, ", beta=", beta), 
       x = expression(mu))


# 作図用のlambdaの値を設定
lambda_line <- seq(0, 4 * lambda_truth, by = 0.00001)

# lambdaの事前分布を計算:式(2.56)
prior_lambda_df <- tibble(
  lambda = lambda_line, # x軸の値
  ln_C_gam = a * log(b) - lgamma(a), # 正規化項(対数)
  density = exp(ln_C_gam + (a - 1) * log(lambda) - b * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)

# lambdaの事前分布を作図
ggplot(prior_lambda_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事前分布
  labs(title = "Gamma Distribution", 
       subtitle = paste0("a=", a, ", b=", b))


### 事後分布(ガウス分布とスチューデントのt分布)の計算 -----

# muの事後分布のパラメータを計算:式(3.83)
beta_hat <- N + beta
m_hat <- (sum(x_n) + beta * m) / beta_hat

# lambdaの事後分布のパラメータを計算:式(3.88)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * (sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) + b

# lambdaの期待値を計算:式(2.59)
E_lambda_hat <- a_hat / b_hat


# muの事後分布を計算:式(2.64)
posterior_mu_df <- tibble(
  mu = mu_line, # x軸の値
  density = dnorm(
    mu, mean = m_hat, sd = sqrt(1 / beta_hat / E_lambda_hat)
  ) # 確率密度
)

# muの事後分布を作図
ggplot(posterior_mu_df, aes(mu, density)) + 
  geom_line(color = "#56256E") + # muの事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真のmu
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", m_hat=", round(m_hat, 1), ", beta_hat=", beta_hat), 
       x = expression(mu))


# lambdaの事後分布を計算
posterior_lambda_df <- tibble(
  lambda = lambda_line, # x軸の値
  ln_C_gam = a_hat * log(b_hat) - lgamma(a_hat), # 正規化項(対数)
  density = exp(ln_C_gam + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a_hat, rate = b_hat) # 確率密度
)

# lambdaの事後分布を作図
ggplot(posterior_lambda_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真のlambda
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", round(b_hat, 1)), 
       x = expression(lambda))


### 予測分布()の計算 -----

# 予測分布のパラメータを計算:式(3.95')
mu_s_hat <- m_hat
lambda_s_hat <- beta_hat * a_hat / (1 + beta_hat) / b_hat
nu_s_hat <- 2 * a_hat
#mu_s_hat <- (sum(x_n) + beta * m) / (N + beta)
#tmp_lambda_numer <- (N + beta) * (N / 2 + a)
#tmp_lambda_denom <- (N + 1 + beta) * ((sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) / 2 + b)
#lambda_s_hat <- tmp_lambda_numer / tmp_lambda_denom
#nu_s_hat <- N + 2 * a

# 予測分布を計算:式(3.76)
predict_df <- tibble(
  x = x_line, # x軸の値
  ln_C_St = lgamma(0.5 * (nu_s_hat + 1)) - lgamma(0.5 * nu_s_hat), # 正規化項(対数)
  ln_term1 = 0.5 * log(lambda_s_hat / pi / nu_s_hat), 
  ln_term2 = - 0.5 * (nu_s_hat + 1) * log(1 + lambda_s_hat / nu_s_hat * (x - mu_s_hat)^2), 
  density = exp(ln_C_St + ln_term1 + ln_term2) # 確率密度
)

# 予測分布を作図
ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = density), 
            color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Student's t Distribution", 
       subtitle = paste0("N=", N, ", mu=", round(mu_s_hat, 1), 
                         ", lambda=", round(lambda_s_hat, 3), ", nu=", nu_s_hat))


# ・アニメーション --------------------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(gganimate)


### 推論処理 -----

# 真のパラメータを指定
mu_truth <- 25
lambda_truth <- 0.01

# muの事前分布のパラメータを指定
m <- 0
beta <- 1

# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1

# lambdaの期待値を計算:式(2.59)
E_lambda <- a / b


# 作図用のmuの値を設定
mu_line <- seq(mu_truth - 30, mu_truth + 30, by = 0.01)

# muの事前分布を計算:式(2.64)
posterior_mu_df <- tibble(
  mu = mu_line, # x軸の値
  density = dnorm(mu, mean = m, sd = sqrt(1 / beta / E_lambda)), # 確率密度
  label = as.factor(paste0("N=", 0, ", m=", round(m, 1), ", beta=", beta)) # パラメータ
)


# 作図用のlambdaの値を設定
lambda_line <- seq(0, 4 * lambda_truth, by = 0.00001)

# lambdaの事前分布を計算:式(2.56)
posterior_lambda_df <- tibble(
  lambda = lambda_line, # x軸の値
  density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
  label = as.factor(paste0("N=", 0, ", a=", a, ", b=", round(b, 1))) # パラメータ
)


# 初期値による予測分布のパラメータを計算:式(3.95)
mu_s <- m
lambda_s <- beta * a / (1 + beta) / b
nu_s <- 2 * a

# 作図用のxの値を設定
x_line <- seq(
  mu_truth - 4 * sqrt(1 / lambda_truth), 
  mu_truth + 4 * sqrt(1 / lambda_truth), 
  by = 0.1
)

# 初期値による予測分布を計算:式(3.95)
predict_df <- tibble(
  x = x_line, # x軸の値
  ln_C_St = lgamma(0.5 * (nu_s + 1)) - lgamma(0.5 * nu_s), # 正規化項(対数)
  ln_term1 = 0.5 * log(lambda_s / pi / nu_s), 
  ln_term2 = - 0.5 * (nu_s + 1) * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
  density = exp(ln_C_St + ln_term1 + ln_term2), # 確率密度
  label = as.factor(
    paste0(
      "N=", 0, ", mu_s=", round(mu_s, 1), ", lambda_s=", round(lambda_s, 3), ", nu_s=", nu_s
    )
  ) # パラメータ
)


# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を初期化
x_n <- rep(0, N)

# ベイズ推論
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu_truth, sd = sqrt(1 / lambda_truth))
  
  # muの事後分布のパラメータを更新:式(3.83)
  beta_old <- beta
  beta <- 1 + beta
  m_old <- m
  m <- (x_n[n] + beta_old * m) / beta
  
  # lambdaの事後分布のパラメータを更新:式(3.88)
  a <- 1 / 2 + a
  b <- (x_n[n]^2 + beta_old * m_old^2 - beta * m^2) / 2 + b
  
  # lambdaの期待値を計算:式(2.59)
  E_lambda <- a / b
  
  # muの事後分布を計算:式(2.64)
  tmp_posterior_mu_df <- tibble(
    mu = mu_line, # x軸の値
    density = dnorm(mu, mean = m, sd = sqrt(1 / beta / E_lambda)), # 確率密度
    label = as.factor(
      paste0("N=", n, ", m_hat=", round(m, 1), ", beta_hat=", beta)
    ) # パラメータ
  )
  
  # lambdaの事後分布を計算:式(2.56)
  tmp_posterior_lambda_df <- tibble(
    lambda = lambda_line, # x軸の値
    density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    label = as.factor(
      paste0("N=", n, ", a_hat=", a, ", b_hat=", round(b, 1))
    ) # パラメータ
  )
  
  # 予測分布のパラメータを更新:式(3.95)
  mu_s <- m
  lambda_s <- beta * a / (1 + beta) / b
  nu_s <- 2 * a
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = x_line, # x軸の値
    ln_C_St = lgamma(0.5 * (nu_s + 1)) - lgamma(0.5 * nu_s), # 正規化項(対数)
    ln_term1 = 0.5 * log(lambda_s / pi / nu_s), 
    ln_term2 = - 0.5 * (nu_s + 1) * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
    density = exp(ln_C_St + ln_term1 + ln_term2), # 確率密度
    label = as.factor(
      paste0(
        "N=", n, ", mu_s_hat=", round(mu_s, 1), ", lambda_s_hat=", round(lambda_s, 3), ", nu_s_hat=", nu_s
      )
    ) # パラメータ
  )
  
  # 推論結果を結合
  posterior_mu_df <- rbind(posterior_mu_df, tmp_posterior_mu_df)
  posterior_lambda_df <- rbind(posterior_lambda_df, tmp_posterior_lambda_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### 作図処理 -----

# muの事後分布を作図
posterior_mu_graph <- ggplot(posterior_mu_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # muの事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真のmu
  transition_manual(label) + # フレーム
  labs(title = "Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu))

# gif画像を出力
animate(posterior_mu_graph, nframes = N + 1, fps = 10)


# lambdaの事後分布を作図
posterior_lambda_graph <- ggplot(posterior_lambda_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真のlambda
  transition_manual(label) + # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda))

# gif画像を出力
animate(posterior_lambda_graph, nframes = N + 1, fps = 10)


# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_line, # x軸の値
  density = dnorm(x = x, mean = mu_truth, sd = sqrt(1 / lambda_truth)) # 確率密度
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


