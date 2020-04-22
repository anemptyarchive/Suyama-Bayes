
# 4.3.3 ポアソン混合モデルにおける変分推論 -------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)


# 真の観測モデルの設定 ----------------------------------------------------------------

# (観測)データ数を指定
N <- 100

# 真のパラメータを指定
lambda_truth <- c(5, 25)
pi_truth <- c(0.3, 0.7)

# クラスタ数
K <- length(lambda_truth)

# クラスタ(潜在変数)を生成
s_nk <- rmultinom(n =  N, size = 1, prob = pi_truth) %>% 
  t()

# (観測)データXを生成
x_n <- rpois(n = N, lambda = apply(lambda_truth^t(s_nk), 2, prod))

# 観測データを確認
summary(x_n)
tibble(x = x_n) %>% 
  ggplot(aes(x = x)) + 
    geom_bar(fill = "#56256E") + 
    labs(title = "Histogram")


# パラメータの設定 ----------------------------------------------------------------

# 試行回数
Iter <- 50

# ハイパーパラメータa,bの初期値を指定
a <- 1
b <- 1

# lambda_k(の期待値)の初期値をランダムに設定
tmp_lambda <- seq(0, 1, by = 0.01) %>% 
  sample(size = K, replace = TRUE)
E_lambda_k    <- tmp_lambda / sum(tmp_lambda) # 正規化
E_ln_lambda_k <- log(E_lambda_k) # 対数をとる

# ハイパーパラメータalphaの初期値を指定
alpha_k <- rep(2, K)

# pi(の対数をとった期待値)の初期値をランダムに設定
tmp_pi <- seq(0, 1, by = 0.01) %>% 
  sample(size = K, replace = TRUE)
E_ln_pi_k <- tmp_pi / sum(tmp_pi) %>% # 正規化
  log() # 対数をとる


# 変分推論 --------------------------------------------------------------------

# 受け皿を用意
eta_nk <- matrix(0, nrow = N, ncol = K)
E_s_nk <- matrix(0, nrow = N, ncol = K)
hat_a_k <- rep(0, K)
hat_b_k <- rep(0, K)

# 推移の確認用
trace_a <- matrix(0, nrow = Iter + 1, ncol = K)
trace_b <- matrix(0, nrow = Iter + 1, ncol = K)
trace_alpha <- matrix(0, nrow = Iter + 1, ncol = K)
# 初期値を代入
trace_a[1, ] <- a
trace_b[1, ] <- b
trace_alpha[1, ] <- alpha_k

for(i in 1:Iter) {
  
  for(n in 1:N) {
    
    # パラメータeta_nを計算:式(4.51)
    tmp_eta <- exp(x_n[n] * E_ln_lambda_k - E_lambda_k + E_ln_pi_k)
    eta_nk[n, ] <- tmp_eta / sum(tmp_eta) # 正規化
    
    # s_nkの期待値を計算:式(4.59)
    E_s_nk[n, ] <- eta_nk[n, ]
  }
  
  for(k in 1:K) {
    
    # ハイパーパラメータa_hat,b_hatを計算:式(4.55)
    hat_a_k[k] <- sum(E_s_nk[, k] * x_n) + a
    hat_b_k[k] <- sum(E_s_nk[, k]) + b
    
    # (対数をとった)lambda_kの期待値を計算:式(4.60),(4.61)
    E_lambda_k[k] <- hat_a_k[k] / hat_b_k[k]
    E_ln_lambda_k[k] <- digamma(hat_a_k[k]) - log(hat_b_k[k])
  }
  
  # ハイパーパラメータalpha_hatを計算:式(4.58)
  hat_alpha_k <- apply(E_s_nk, 2, sum) + alpha_k
  
  # 対数をとったpi_kの期待値を計算:式(4.62)
  E_ln_pi_k <- digamma(hat_alpha_k) - digamma(sum(hat_alpha_k))
  
  # 推移の確認用に推定結果を保存
  trace_a[i + 1, ] <- hat_a_k
  trace_b[i + 1, ] <- hat_b_k
  trace_alpha[i + 1, ] <- hat_alpha_k
  
}


# 結果の確認 -------------------------------------------------------------------

## lambdaの近似事後分布
# データフレームを作成
lambda_df <- tibble()
for(k in 1:K) {
  # データフレームに変換
  tmp_lambda_df <- tibble(
    lambda = seq(0, max(x_n), by = 0.01), 
    density = dgamma(lambda, shape = hat_a_k[k], rate = hat_b_k[k]), 
    cluster = as.factor(k)
  )
  # 結合
  lambda_df <- rbind(lambda_df, tmp_lambda_df)
}

# 作図
ggplot(lambda_df, aes(lambda, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = lambda_truth, color = "pink", linetype = "dashed") + # 垂直線
  labs(title = "Poisson mixture model:variational inference", 
       subtitle = paste0("a_hat=(", paste0(round(hat_a_k, 1), collapse = ", "), 
                         "), b_hat=(", paste0(round(hat_b_k, 1), collapse = ", "), ")")) # ラベル


## piの近似事後分布(K=2のときのみ可)
# データフレームを作成
pi_df <- tibble()
for(k in 1:K) {
  # データフレームに変換
  tmp_pi_df <- tibble(
    pi = seq(0, 1, by = 0.01), 
    density = dbeta(pi, shape1 = hat_alpha_k[k], shape2 = hat_alpha_k[2 / k]), 
    cluster = as.factor(k)
  )
  # 結合
  pi_df <- rbind(pi_df, tmp_pi_df)
}

# 作図
ggplot(pi_df, aes(pi, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = pi_truth, color = "pink", linetype = "dashed") + # 垂直線
  labs(title = "Poisson mixture model:variational inference", 
       subtitle = paste0("alpha_hat=(", paste0(round(hat_alpha_k, 1), collapse = ", "), ")")) # ラベル



# パラメータの推移の確認 -------------------------------------------------------------

## lambdaの近似事後分布
# 作図用のデータフレームを作成
trace_lambda_long <- tibble()
for(i in 1:(Iter + 1)) {
  for(k in 1:K) {
    # データフレームに変換
    tmp_lambda_df <- tibble(
      lambda = seq(0, max(x_n), by = 0.01), 
      density = dgamma(lambda, shape = trace_a[i, k], rate = trace_b[i, k]), 
      cluster = as.factor(k), 
      Iteration = i - 1
    )
    # 結合
    trace_lambda_long <- rbind(trace_lambda_long, tmp_lambda_df)
  }
}

# 作図
graph_lambda <- ggplot(trace_lambda_long, aes(lambda, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = lambda_truth, color = "pink", linetype = "dashed") + # 垂直線
  transition_manual(Iteration) + # フレーム
  labs(title = "Poisson mixture model:variational inference", 
       subtitle = "i={current_frame}") # ラベル
  
# gif画像を作成
animate(graph_lambda, nframes = Iter + 1, fps = 5)


## piの近似事後分布(K=2のときのみ可)
# 作図用のデータフレームを作成
trace_pi_long <- tibble()
for(i in 1:(Iter + 1)) {
  for(k in 1:K) {
    # データフレームに変換
    tmp_pi_df <- tibble(
      pi = seq(0, 1, by = 0.01), 
      density = dbeta(pi, shape1 = trace_alpha[i, k], shape2 = trace_alpha[i, 2 / k]), 
      cluster = as.factor(k), 
      Iteration = i - 1
    )
    # 結合
    trace_pi_long <- rbind(trace_pi_long, tmp_pi_df)
  }
}

# 作図
graph_pi <- ggplot(trace_pi_long, aes(pi, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = pi_truth, color = "pink", linetype = "dashed") + # 垂直線
  transition_manual(Iteration) + # フレーム
  labs(title = "Poisson mixture model:variational inference", 
       subtitle = "i={current_frame}") # ラベル

# gif画像を作成
animate(graph_pi, nframes = Iter + 1, fps = 5)



# ハイパーパラメータの推移の確認 ---------------------------------------------------------

## lambdaのパラメータa
# データフレームに変換
trace_a_wide <- cbind(
  as.data.frame(trace_a), 
  Iteration = 1:(Iter + 1)
)

# long型に変換
trace_a_long <- pivot_longer(
  trace_a_wide, 
  cols = -Iteration, 
  names_to = "cluster", 
  names_prefix = "V", 
  names_ptypes = list(cluster = factor()), 
  values_to = "value"
)

# 作図
ggplot(trace_a_long, aes(Iteration, value, color = cluster)) + 
  geom_line() + 
  labs(title = "Poisson mixture model:variational inference", 
       subtitle = expression(hat(a)))


## lambdaのパラメータb
# データフレームに変換
trace_b_wide <- cbind(
  as.data.frame(trace_a), 
  Iteration = 1:(Iter + 1)
)

# long型に変換
trace_b_long <- pivot_longer(
  trace_b_wide, 
  cols = -Iteration, 
  names_to = "cluster", 
  names_prefix = "V", 
  names_ptypes = list(cluster = factor()), 
  values_to = "value"
)

# 作図
ggplot(trace_b_long, aes(Iteration, value, color = cluster)) + 
  geom_line() + 
  labs(title = "Poisson mixture model:variational inference", 
       subtitle = expression(hat(b)))


## piのパラメータ
# データフレームに変換
trace_alpha_wide <- cbind(
  as.data.frame(trace_alpha), 
  Iteration = 1:(Iter + 1)
)

# long型に変換
trace_alpha_long <- pivot_longer(
  trace_alpha_wide, 
  cols = -Iteration, 
  names_to = "cluster", 
  names_prefix = "V", 
  names_ptypes = list(cluster = factor()), 
  values_to = "value"
)

# 作図
ggplot(trace_alpha_long, aes(Iteration, value, color = cluster)) + 
  geom_line() + 
  labs(title = "Poisson mixture model:variational inference", 
       subtitle = expression(hat(alpha)))


