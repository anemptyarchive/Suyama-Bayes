
# ch4.4.2 ガウス混合モデルにおけるギブスサンプリング -------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvnfast)


# モデルの設定 ------------------------------------------------------------------

# 次元数:(固定)
D <- 2

# クラスタ数を指定
K <- 3

# 真の平均パラメータを指定
mu_true_kd <- matrix(
  c(0, 4, 
    -5, -5, 
    5, -2.5), nrow = K, ncol = D, byrow = TRUE
)

# 真の分散共分散行列を指定
sigma2_true_ddk <- array(
  c(8, 0, 0, 8, 
    4, -2.5, -2.5, 4, 
    6.5, 4, 4, 6.5), dim = c(D, D, K)
)

# 真の混合比率を指定
pi_true_k <- c(0.5, 0.2, 0.3)


# 観測データの生成 ----------------------------------------------------------------

# データ数を指定
N <- 250

# 観測データの真のクラスタを生成
s_true_nk <- rmultinom(n = N, size = 1, prob = pi_true_k) %>% 
  t()
res_s <- which(t(s_true_nk) == 1, arr.ind = TRUE)
s_true_n <- res_s[, "row"]

# 観測データを生成
x_nd <- matrix(0, nrow = N, ncol = D)
for(n in 1:N) {
  k <- s_true_n[n] # クラスタを取得
  x_nd[n, ] = mvnfast::rmvn(
    n = 1, mu = mu_true_kd[k, ], sigma = sigma2_true_ddk[, , k]
  )
}


# 作図用の格子状の点を生成
x_line <- seq(-10, 10, by = 0.1) # 描画範囲
point_df <- tibble(
  x_1 = rep(x_line, times = length(x_line)), 
  x_2 = rep(x_line, each = length(x_line))
)

# 作図用のデータフレームを作成
model_true_df <- tibble()
sample_df <- tibble()
for(k in 1:K) {
  # 真の観測モデルを計算
  tmp_model_df <- cbind(
    point_df, 
    density = mvnfast::dmvn(
      X = as.matrix(point_df), mu = mu_true_kd[k, ], sigma = sigma2_true_ddk[, , k]
    ), 
    cluster = as.factor(k)
  )
  model_true_df <- rbind(model_true_df, tmp_model_df)
  
  # 観測データのデータフレーム
  k_idx <- which(s_true_n == k)
  tmp_sample_df <- tibble(
    x_1 = x_nd[k_idx, 1], 
    x_2 = x_nd[k_idx, 2], 
    cluster = as.factor(k)
  )
  sample_df <- rbind(sample_df, tmp_sample_df)
}

# 真の観測モデルを作図
ggplot() + 
  geom_contour(data = model_true_df, aes(x_1, x_2, z = density, color = cluster)) + # 真の観測モデル
  geom_point(data = sample_df, aes(x_1, x_2, color = cluster)) + # 真の観測データ
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0('K=', K, ', N=', N), 
       x = expression(x[1]), y = expression(x[2]))


# 事前分布の設定 --------------------------------------------------------------------

# 平均の事前分布のパラメータを指定
beta <- 1
m_d <- rep(0, D)

# 分散共分散行列の事前分布のパラメータを指定
nu <- D
w_dd <- diag(D) * 0.05

# 混合比率の事前分布のパラメータを指定
alpha_k <- rep(1, K)


# 推論処理 -----------------------------------------------------------------

# クラスタの初期値を生成
s_nk <- rmultinom(n = N, size = 1, prob = alpha_k / sum(alpha_k)) %>% 
  t()


# 混合比率の事後分布の初期値を計算
alpha_hat_k <- colSums(s_nk) + alpha_k

# 平均の事後分布の初期値を計算
beta_hat_k <- colSums(s_nk) + beta
m_hat_kd <- matrix(0, nrow = K, ncol = D)
for(k in 1:K) {
  m_hat_kd[k, ] <- (colSums(s_nk[, k] * x_nd) + beta * m_d) / beta_hat_k[k]
}

# 分散共分散行列の事後分布の初期値を計算
nu_hat_k <- colSums(s_nk) + nu
w_hat_ddk <- array(0, dim = c(D, D, K))
for(k in 1:K) {
  nu_hat_k[k] <- sum(s_nk[, k]) + nu
  tmp_w1_dd <- t(s_nk[, k] * x_nd) %*% x_nd
  tmp_w2_dd <- beta * matrix(m_d) %*% t(m_d)
  tmp_w3_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
  w_hat_ddk[, , k] <- solve(
    tmp_w1_dd + tmp_w2_dd - tmp_w3_dd + solve(w_dd)
  )
}


# 試行回数を指定
MaxIter <- 100

# 途中計算に用いる項の受け皿を作成
term_st_k <- rep(0, K)

# 推移の確認用の受け皿を作成
trace_E_mu_ikd <- array(0, dim = c(MaxIter+1, K, D))
trace_E_lambda_iddk <- array(0, dim = c(MaxIter+1, D, D, K))
trace_E_s_ink <- array(0, dim = c(MaxIter, N, K))
trace_E_mu_ikd[1, , ] <- m_hat_kd
trace_E_lambda_iddk[1, , , ] <- rep(nu_hat_k, each = D * D) * w_hat_ddk

# 周辺化ギブスサンプリング
for(i in 1:MaxIter) {
  for(n in 1:N) {
    
    # 超パラメータからx_nに関する統計量を除去
    k <- which(s_nk[n, ] == 1)
    beta_hat_k[k] <- beta_hat_k[k] - s_nk[n, k]
    m_hat_kd[k, ] <- m_hat_kd[k, ] - x_nd[n, ] / beta_hat_k[k]
    nu_hat_k[k] <- nu_hat_k[k] - s_nk[n, k]
    term_w1_dd <- matrix(x_nd[n, ]) %*% t(x_nd[n, ])
    term_w2_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
    w_hat_ddk[, , k] <- solve(
      solve(w_hat_ddk[, , k]) - term_w1_dd + term_w2_dd
    )
    alpha_hat_k[k] <- alpha_hat_k[k] - s_nk[n, k]
    
    # t分布のパラメータを計算
    nu_st_hat_k <- 1 - D + nu_hat_k
    term_lmd_k <- nu_st_hat_k * beta_hat_k / (1 + beta_hat_k)
    lambda_st_hat_ddk <- w_hat_ddk * array(rep(term_lmd_k, each = D * D), dim = c(D, D, K))
    
    # クラスタのサンプリング確率を計算
    term_st_nu_k <- - 0.5 * (1 + nu_hat_k)
    for(k in 1:K) {
      term_st_x_d <- x_nd[n, ] - m_hat_kd[k, ]
      term_st_k[k] <- t(term_st_x_d) %*% lambda_st_hat_ddk[, , k] %*% matrix(term_st_x_d)
    }
    term_st_ln_k <- log(1 + term_st_k / nu_st_hat_k)
    term_p_s_k <- exp(term_st_nu_k * term_st_ln_k)
    eta_k <- alpha_hat_k / sum(alpha_hat_k)
    tmp_p_s_k <- term_p_s_k * eta_k
    p_s_k <- tmp_p_s_k / sum(tmp_p_s_k)
    
    # クラスタをサンプル
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = p_s_k) %>% 
      as.vector()
    
    
    # 超パラメータからx_nに関する統計量を加算
    k <- which(s_nk[n, ] == 1)
    beta_hat_k[k] <- beta_hat_k[k] + s_nk[n, k]
    m_hat_kd[k, ] <- m_hat_kd[k, ] + x_nd[n, ] / beta_hat_k[k]
    nu_hat_k[k] <- nu_hat_k[k] + s_nk[n, k]
    term_w1_dd <- matrix(x_nd[n, ]) %*% t(x_nd[n, ])
    term_w2_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
    w_hat_ddk[, , k] <- solve(
      solve(w_hat_ddk[, , k]) + term_w1_dd - term_w2_dd
    )
    alpha_hat_k[k] <- alpha_hat_k[k] + s_nk[n, k]
  }
}

i


# 作図用のデータフレームを作成
model_df <- tibble()
sample_df <- tibble()
for(k in 1:K) {
  # 近似事後分布を計算
  tmp_model_df <- cbind(
    point_df, 
    density = mvnfast::dmvn(
      as.matrix(point_df), mu = m_hat_kd[k, ], sigma = solve(nu_hat_k[k] * w_hat_ddk[, , k])
    ), 
    cluster = as.factor(k)
  )
  model_df <- rbind(model_df, tmp_model_df)
  
  # 観測データのクラスタを抽出
  k_idx <- which(s_nk[, k] == 1)
  tmp_sample_df <- tibble(
    x_1 = x_nd[k_idx, 1], 
    x_2 = x_nd[k_idx, 2], 
    cluster = as.factor(k)
  )
  sample_df <- rbind(sample_df, tmp_sample_df)
}

# 近似事後分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x_1, x_2, z = density, color = cluster)) + # 近似事後分布
  geom_contour(data = model_true_df, aes(x_1, x_2, z = density, color = cluster), 
               linetype = "dotted", alpha = 0.6) + # 真の観測モデル
  geom_point(data = sample_df, aes(x_1, x_2, color = cluster)) + # 観測データ
  labs(title = "Collapsed Gibbs Sampling", 
       subtitle = paste0('K=', K, ', N=', N, ', iter:', MaxIter), 
       x = expression(x[1]), y = expression(x[2]))

