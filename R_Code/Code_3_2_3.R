
# 3.2.3 ポアソン分布の学習と予測 ---------------------------------------------------------

# 3.2.3 事後分布 --------------------------------------------------------------


# 利用パッケージ
library(tidyverse)


## パラメータの初期値を設定
# 観測モデルのパラメータ
lambda_truth <- 4

# 事前分布のパラメータ
a <- 1
b <- 1

# 試行回数
N <- 50


# ポアソン分布に従うデータを生成
x_n <- rpois(n = N ,lambda = lambda_truth)

# 観測データを確認
x_n_df <- x_n %>% 
  table() %>% 
  as_tibble()
colnames(x_n_df) <- c("n", "x")
x_n_df$n <- as.numeric(x_n_df$n)
x_n_df


# 事後分布のパラメータを計算
a_hat <- sum(x_n) + a
b_hat <- N + b

# 事後分布を計算
posterior_df <- tibble(
  lambda = seq(0, 2 * lambda_truth, by = 0.001),  # 作図用の値
  C_G = a_hat * log(b_hat) - lgamma(a_hat),   # 正規化項(対数)
  density = exp(C_G + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
)


# 作図
ggplot(posterior_df, aes(lambda, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", b_hat), 
       x = expression(lambda)) # ラベル



# 3.2.3 予測分布 --------------------------------------------------------------


# 予測分布のパラメータを計算
r_hat <- a_hat
p_hat <- 1 / (b_hat + 1)

#r_hat <- sum(x_n) + a
#p_hat <- 1 / (N + 1 + b)

# 予測分布を計算
predict_df <- tibble(
  x = seq(0, 4 * lambda_truth),  # 作図用の値
  C_NB = lgamma(x + r_hat) - lgamma(x + 1) - lgamma(r_hat), # 正規化項(対数)
  prob = exp(C_NB + r_hat * log(1 - p_hat) + x * log(p_hat)) # 確率
)


# 作図
ggplot(predict_df, aes(x, prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "#56256E") + # 棒グラフ
  labs(title = "Negative Binomial Distribution", 
       subtitle = paste0("N=", N, ", r_hat=", r_hat, ", p_hat=", round(p_hat, 2))) # ラベル


# 3.2.3 gif ---------------------------------------------------------------


# 利用パッケージ
library(tidyverse)
library(gganimate)


## パラメータの初期値を設定
# 観測モデルのパラメータ
lambda_truth <- 4

# 事前分布のパラメータ
a <- 1
b <- 1

# 試行回数
N <- 50


# 事前分布を計算
posterior_df <- tibble(
  lambda = seq(0, 2 * lambda_truth, by = 0.001),  # 作図用の値
  C_G = a * log(b) - lgamma(a),   # 正規化項(対数)
  density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
  N = 0 # 試行回数
)


# 初期値による予測分布のパラメータを計算
r <- a
p <- 1 / (b + 1)

# 初期値による予測分布を計算
predict_df <- tibble(
  x = seq(0, 4 * lambda_truth),  # 作図用の値
  C_NB = lgamma(x + r) - lgamma(x + 1) - lgamma(r), # 正規化項(対数)
  prob = exp(C_NB + r * log(1 - p) + x * log(p)), # 確率
  N = 0 # 試行回数
)


# パラメータを推定
x_n <- rep(0, N) # 受け皿
for(n in 1:N){
  
  # ポアソン分布に従うデータを生成
  x_n[n] <- rpois(n = 1 ,lambda = lambda_truth)
  
  
  # ハイパーパラメータを更新
  a <- sum(x_n[n] * 1) + a
  b <- 1 + b
  
  # 事後分布を推定
  tmp_posterior_df <- tibble(
    lambda = seq(0, 2 * lambda_truth, by = 0.001),  # 作図用のlamndaの値
    C_G = a * log(b) - lgamma(a),   # 正規化項(対数)
    density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
    N = n # 試行回数
  )
  
  
  # 予測分布のパラメータを計算
  r <- a
  p <- 1 / (b + 1)
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = seq(0, 4 * lambda_truth),  # 作図用の値
    C_NB = lgamma(x + r) - lgamma(x + 1) - lgamma(r), # 正規化項(対数)
    prob = exp(C_NB + r * log(1 - p) + x * log(p)), # 確率
    N = n # 試行回数
  )
  
  
  # 結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
table(x_n)


## 事後分布
# 作図
posterior_graph <- ggplot(posterior_df, aes(lambda, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  transition_manual(N) + # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(lambda)) # ラベル

# 描画
animate(posterior_graph)


## 予測分布
# 作図
predict_graph <- ggplot(predict_df, aes(x, prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "#56256E") + # 棒グラフ
  transition_manual(N) + # フレーム
  labs(title = "Negative Binomial Distribution", 
       subtitle = "N= {current_frame}") # ラベル

# 描画
animate(predict_graph)


