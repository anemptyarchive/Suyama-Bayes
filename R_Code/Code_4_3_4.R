
# 4.3.3 ポアソン混合モデルにおける崩壊型ギブスサンプリング -------------------------------------------------

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

# 試行回数を指定
MaxIter <- 50

# ハイパーパラメータa,bを指定
a <- 1
b <- 1
alpha_k <- rep(2, K)

# 潜在変数Sの初期値をランダムに設定
s_nk <- rmultinom(n = N, size = 1, prob = rep(1, K)) %>% 
  t()

# Xに関する統計量を計算
hat_a_k <- apply(s_nk * x_n, 2, sum) + a
hat_b_k <- apply(s_nk, 2, sum) + b
hat_alpha_k <- apply(s_nk, 2, sum) + alpha_k


# 崩壊型ギブスサンプリング ------------------------------------------------------------

# 受け皿を用意
eta_k.n <- rep(0, K)
hat_r_k <- rep(0, K)
hat_p_k <- rep(0, K)


# ハイパーパラメータの推定値の推移の確認用
trace_a <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_b <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_alpha <- matrix(0, nrow = MaxIter + 1, ncol = K)
# 初期値を代入
trace_a[1, ] <- hat_a_k.n
trace_b[1, ] <- hat_b_k.n
trace_alpha[1, ] <- hat_alpha_k.n

for(i in 1:MaxIter) {
   
  # 初期化
  new_a_k <- rep(0, K)
  new_b_k <- rep(0, K)
  new_alpha_k <- rep(0, K)
  
  for(n in 1:N) {
    
    # 更新した統計量を移す
    hat_a_k.n <- hat_a_k
    hat_b_k.n <- hat_b_k
    hat_alpha_k.n <- hat_alpha_k

    
    # x_nに関する統計量を除去:式(4.82),(4.83)
    hat_a_k.n <- hat_a_k.n - s_nk[n, ] * x_n[n]
    hat_b_k.n <- hat_b_k.n - s_nk[n, ]
    hat_alpha_k.n <- hat_alpha_k.n - s_nk[n, ]
    
    for(k in 1:K) {
      
      # ハイパーパラメータr_hat,p_hatを計算:式(4.80)
      hat_r_k[k] <- hat_a_k.n[k]
      hat_p_k[k] <- 1 / (hat_b_k.n[k] + 1)
      
      # :式(4.81)
      C_NB <- lgamma(x_n[n] + hat_r_k[k]) - lgamma(x_n[n] + 1) - lgamma(hat_r_k[k]) # 正規化項(対数)
      p_x_n <- exp(C_NB + hat_r_k[k] * log(1 - hat_p_k[k]) + x_n[n] * log(hat_p_k[k]))
    }
    
    # パラメータeta_nを計算:式(4.75)
    eta_k.n <- hat_alpha_k.n / sum(hat_alpha_k.n)
    
    # 潜在変数s_nをサンプル:式(4.74)
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = eta_nk[n, ]) %>% 
      as.vector()
    
    # x_nに関する統計量を追加:式(4.82),(4.83)
    new_a_k <- hat_a_k + s_nk[n, ] * x_n[n]
    new_b_k <- hat_b_k + s_nk[n, ]
    new_alpha_k <- hat_alpha_k + s_nk[n, ]
  }
  
  #
  # 更新
  hat_a_k <- new_a_k
  hat_b_k <- new_b_k
  hat_alpha_k <- new_alpha_k
  
  # 推移の確認用に推定結果を保存
  trace_a[i + 1, ] <- hat_a_k
  trace_b[i + 1, ] <- hat_b_k
  trace_alpha[i + 1, ] <- hat_alpha_k
}


# 結果の確認 -------------------------------------------------------------------

## lambdaの近似事後分布
# 作図用のデータフレームを作成
lambda_df <- tibble()
for(k in 1:K) {
  # データフレームに変換
  tmp_lambda_df <- tibble(
    lambda = seq(0, max(x_n), by = 0.01), 
    density = dgamma(lambda, shape = hat_a_k.n[k], rate = hat_b_k.n[k]), 
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("a_hat=(", paste0(round(hat_a_k.n, 1), collapse = ", "), 
                         "), b_hat=(", paste0(round(hat_b_k.n, 1), collapse = ", "), ")")) # ラベル


## piの近似事後分布(K=2のときのみ可能)
# 作図用のデータフレームを作成
pi_df <- tibble()
for(k in 1:K) {
  # データフレームに変換
  tmp_pi_df <- tibble(
    pi = seq(0, 1, by = 0.01), 
    density = dbeta(pi, shape1 = hat_alpha_k.n[k], shape2 = hat_alpha_k.n[2 / k]), 
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = paste0("alpha_hat=(", paste0(round(hat_alpha_k.n, 1), collapse = ", "), ")")) # ラベル



# パラメータの推移の確認 -------------------------------------------------------------

## lambdaの近似事後分布
# 作図用のデータフレームを作成
trace_lambda_long <- tibble()
for(i in 1:(MaxIter + 1)) {
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = "i={current_frame}") # ラベル

# gif画像を作成
animate(graph_lambda, nframes = MaxIter + 1, fps = 5)


## piの近似事後分布(K=2のときのみ可)
# 作図用のデータフレームを作成
trace_pi_long <- tibble()
for(i in 1:(MaxIter + 1)) {
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = "i={current_frame}") # ラベル

# gif画像を作成
animate(graph_pi, nframes = MaxIter + 1, fps = 5)



# ハイパーパラメータの推移の確認 ---------------------------------------------------------

## lambdaのパラメータa
# データフレームに変換
trace_a_wide <- cbind(
  as.data.frame(trace_a), 
  Iteration = 1:(MaxIter + 1)
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = expression(hat(a)))


## lambdaのパラメータb
# データフレームに変換
trace_b_wide <- cbind(
  as.data.frame(trace_a), 
  Iteration = 1:(MaxIter + 1)
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = expression(hat(b)))


## piのパラメータ
# データフレームに変換
trace_alpha_wide <- cbind(
  as.data.frame(trace_alpha), 
  Iteration = 1:(MaxIter + 1)
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
  labs(title = "Poisson Mixture Model:Variational Inference", 
       subtitle = expression(hat(alpha)))


