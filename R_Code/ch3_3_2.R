
# 3.3.2 1次元ガウス分布の学習と予測：精度が未知の場合 ------------------------------------------------------------------

# 3.3.2 事後分布 -------------------------------------------------------------


# 利用パッケージ
library(tidyverse)


## パラメータの初期値を指定
# 観測モデルのパラメータ
lambda_truth <- 0.01
mu <- 25

# 事前分布のパラメータ
a <- 1
b <- 1

# 試行回数
N <- 50


# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu, sd = sqrt(lambda_truth^(-1)))

# 観測データを確認
summary(x_n)


# 事後分布のパラメータを計算
a_hat <- N / 2 + a
b_hat <- sum((x_n - mu)^2) / 2 + b

# 事後分布を計算
posterior_df <- tibble(
  lambda = seq(0, 3 * lambda_truth, by = 0.0001),  # 作図用の値
  C_G = a_hat * log(b_hat) - lgamma(a_hat),  # 正規化項(対数)
  density = exp(C_G + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
)


# 作図
ggplot(posterior_df, aes(lambda, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", round(b_hat, 2)), 
       x = expression(lambda)) # ラベル



# 3.3.2 予測分布 --------------------------------------------------------------


# 予測分布のパラメータを計算
mu_s <- mu
lambda_s_hat <- a_hat / b_hat
nu_s_hat <- 2 * a_hat

#lambda_s_hat <- (N + 2 * a) / (sum((x_n - mu)^2) + 2 * b)
#nu_s_hat <- N + 2 * a

# 予測分布を計算
predict_df <- tibble(
  x = seq(-mu_s, 3 * mu_s, by = 0.01),  # 作図用の値
  C_St = lgamma((nu_s_hat + 1) / 2) - lgamma(nu_s_hat / 2),  # 正規化項(対数)
  term1 = log(lambda_s_hat / pi / nu_s_hat) / 2, 
  term2 = - (nu_s_hat + 1) / 2 * log(1 + lambda_s_hat / nu_s_hat * (x - mu_s)^2), 
  density = exp(C_St + term1 + term2) # 確率密度
)


# 作図
ggplot(predict_df, aes(x, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  labs(title = "Student's t Distribution", 
       subtitle = paste0("N=", N, ", mu_s=", mu_s, 
                         ", lambda_s_hat=", round(lambda_s_hat, 2), 
                         ", nu_s_hat=", nu_s_hat)) # ラベル


# 3.3.2 try ---------------------------------------------------------------

predict_df <- tibble(
  x = seq(0, 2 * mu_s, by = 0.01),  # 作図用の値
  t = x - mu_s,  # 中心をズラす
  density = dt(t, df = nu_s_hat) # 確率密度
)


# 3.3.2 gif ---------------------------------------------------------------


# 利用パッケージ
library(tidyverse)
library(gganimate)


## パラメータの初期値を指定
# 観測モデルのパラメータ
lambda_truth <- 0.01
mu <- 25

# 事前分布のパラメータ
a <- 1
b <- 1

# 試行回数
N <- 50


# 事前分布を計算
posterior_df <- tibble(
  lambda = seq(0, 3 * lambda_truth, by = 0.0001),  # 作図用の値
  C_G = a * log(b) - lgamma(a),  # 正規化項(対数)
  density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
  N = 0  # 試行回数
)


# 初期値による予測分布のパラメータを計算
mu_s <- mu
lambda_s <- a / b
nu_s <- 2 * a

# 初期値による予測分布を計算
predict_df <- tibble(
  x = seq(-mu_s, 3 * mu_s, by = 0.01),  # 作図用の値
  C_St = lgamma((nu_s + 1) / 2) - lgamma(nu_s / 2),  # 正規化項(対数)
  term1 = log(lambda_s / pi / nu_s) / 2, 
  term2 = - (nu_s + 1) / 2 * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
  density = exp(C_St + term1 + term2), # 確率密度
  N = 0 # 試行回数
)

# パラメータを推定
x_n <- rep(0, N) # 受け皿
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu, sd = sqrt(lambda_truth^(-1)))
  
  
  # ハイパーパラメータを更新
  a <- 1 / 2 + a
  b <- (x_n[n] - mu)^2 / 2 + b
  
  # 事後分布を計算
  tmp_posterior_df <- tibble(
    lambda = seq(0, 3 * lambda_truth, by = 0.0001),  # 作図用のlambda
    C_G = a * log(b) - lgamma(a),  # 正規化項(対数)
    density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
    N = n  # 試行回数
  )
  
  
  # 予測分布のパラメータを更新
  mu_s <- mu
  lambda_s <- a / b
  nu_s <- 2 * a
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = seq(-mu_s, 3 * mu_s, by = 0.01),  # 作図用の値
    C_St = lgamma((nu_s + 1) / 2) - lgamma(nu_s / 2),  # 正規化項(対数)
    term1 = log(lambda_s / pi / nu_s) / 2, 
    term2 = - (nu_s + 1) / 2 * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
    density = exp(C_St + term1 + term2), # 確率密度
    N = n # 試行回数
  )
  
  
  # 結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


## 事後分布
# 作図
posterior_graph <- ggplot(posterior_df, aes(lambda, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  transition_manual(N) +  # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(lambda)) # ラベル

# 描画
animate(posterior_graph)


## 予測分布
# 作図
predict_graph <- ggplot(predict_df, aes(x, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  transition_manual(N) + # フレーム
  labs(title = "Student's t Distribution", 
       subtitle = "N= {current_frame}") # ラベル

# 描画
animate(predict_graph)


