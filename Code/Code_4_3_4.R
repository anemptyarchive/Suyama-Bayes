
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

# 試行回数
MaxIter <- 50

# 潜在変数Sの初期値をランダムに設定
s_nk <- rmultinom(n = N, size = 1, prob = rep(1, K)) %>% 
  t()

# 崩壊型ギブスサンプリング ------------------------------------------------------------

# 受け皿を用意
hat_a_k.nk <- rep(0, K)
hat_b_k.nk <- rep(0, K)
hat_alpha_k.nk <- rep(0, K)
hat_r_k <- rep(0, K)
hat_p_k <- rep(0, K)

# 
hat_a_k <- apply(s_nk * x_n, 2, sum) + a
hat_b_k <- apply(s_nk, 2, sum) + b
hat_alpha_k <- s_nk + alpha


for(i in 1:MaxIter) {
  
  for(n in 1:N) {
    
    # :式(4.82),(4.83)
    hat_a_k.nk <- hat_a_k.nk - s_nk[n, ] * x_n[n]
    hat_b_k.nk <- hat_b_k.nk - s_nk[n, ]
    hat_alpha_k.nk <- hat_alpha_k.nk - s_nk[n, ]
    
    for(k in 1:K) {
      
      # ハイパーパラメータr_hat,p_hatを計算:式(4.80)
      hat_r_k[k] <- hat_a_k.nk[k]
      hat_p_k[k] <- 1 / (hat_b_k.nk[k] + 1)
      
      # :式(4.81)
      x_n[n] <- rnbinom(n = 1, size = 1, prob = hat_p_k[k], mu = hat_r_k[k])
    }
    
    # パラメータeta_nを計算:式(4.75)
    tmp_eta <- hat_alpha_k.nk
    eta_nk[n, ] <- tmp_eta / sum(tmp_eta)
    
    # 潜在変数s_nをサンプル:式(4.74)
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = eta_nk[n, ]) %>% 
      as.vector()
    
    # :式(4.83)
    hat_a_k.nk <- hat_a_k.nk + s_nk[n, ] * x_n[n]
    hat_b_k.nk <- hat_a_k.nk + s_nk[n, ]
    hat_alpha_k.nk <- hat_alpha_k.nk + s_nk[n, ]
  }
}
