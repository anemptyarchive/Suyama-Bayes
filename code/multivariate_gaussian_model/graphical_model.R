
# 多次元ガウスモデル ------------------------------------------------------------

# chapter 3.4
# グラフィカルモデル表現


# ライブラリの読込 -------------------------------------------------------------

# 利用パッケージ
library(DiagrammeR)
library(DiagrammeRsvg)


# グラフィカルモデルの作図 -----------------------------------------------------

### 平均が未知の場合 -----

# 多次元ガウスモデルのグラフィカルモデルを作図
graph <- DiagrammeR::grViz("
  digraph dot{
    label    = 'multivariate Gaussian model \n unknown mean'
    labelloc = t
    fontsize = 20
    
    graph [rankdir = LR]
    node  [shape = circle, fixedsize = ture, height = 0.6, fontname = 'Times-Italic']
    edge  []
    
    m         [label = <<B>m</B>>]
    lambda_mu [label = <<B>&Lambda;</B>@_{-&mu;}>] ## ( `{-}` は、出力時に添字が重なる対策用の小細工)
    
    mu     [label = <<B>&mu;</B>>]
    lambda [label = <<B>&Lambda;</B>>, style = filled, filledcolor = gray]
    
    subgraph cluster_n{
      label    = 'N'
      fontsize = 14
      
      x [label = <<B>x</B>@_{n}>, style = filled, filledcolor = gray]
    }
    
    {rank = same; mu; lambda}
    
    {m, lambda_mu} -> mu -> x;
    lambda -> x;
  }
")

# グラフを書出
DiagrammeRsvg::export_svg(gv = graph) |> # svgファイルに変換
  charToRaw() |> 
  rsvg::rsvg(height = 600) |> # ビットマップに変換
  png::writePNG(target = "figure/graphical_model/multivariate_gaussian_model_mean.png", dpi = 100) # pngファイルに変換


### 精度が未知の場合 -----

# 多次元ガウスモデルのグラフィカルモデルを作図
graph <- DiagrammeR::grViz("
  digraph dot{
    label    = 'multivariate Gaussian model \n unknown precision'
    labelloc = t
    fontsize = 20
    
    graph [rankdir = LR]
    node  [shape = circle, fixedsize = ture, height = 0.6, fontname = 'Times-Italic']
    edge  []
    
    nu [label = '&nu;']
    w  [label = <<B>W</B>>]
    
    mu     [label = <<B>&mu;</B>>, style = filled, filledcolor = gray]
    lambda [label = <<B>&Lambda;</B>>]
    
    subgraph cluster_n{
      label    = 'N'
      fontsize = 14
      
      x [label = <<B>x</B>@_{n}>, style = filled, filledcolor = gray]
    }
    
    {rank = same; mu; lambda}
    
    mu -> x;
    {nu, w} -> lambda -> x;
    mu -> lambda [arrowhead = none, penwidth = 0]; ## (エッジ設定は、ノード位置の共通化用の小細工)
  }
")

# グラフを書出
DiagrammeRsvg::export_svg(gv = graph) |> # svgファイルに変換
  charToRaw() |> 
  rsvg::rsvg(height = 600) |> # ビットマップに変換
  png::writePNG(target = "figure/graphical_model/multivariate_gaussian_model_precision.png", dpi = 100) # pngファイルに変換


### 平均と精度が未知の場合 -----

# 多次元ガウスモデルのグラフィカルモデルを作図
graph <- DiagrammeR::grViz("
  digraph dot{
    label    = 'multivariate Gaussian model \n unknown mean and precision'
    labelloc = t
    fontsize = 20
    
    graph [rankdir = LR]
    node  [shape = circle, fixedsize = ture, height = 0.6, fontname = 'Times-Italic']
    edge  []
    
    m    [label = <<B>m</B>>]
    beta [label = '&beta;']
    nu   [label = '&nu;']
    w    [label = <<B>W</B>>]
    
    mu     [label = <<B>&mu;</B>>]
    lambda [label = <<B>&Lambda;</B>>]
    
    subgraph cluster_n{
      label    = 'N'
      fontsize = 14
      
      x [label = 'x@_{n}', style = filled, filledcolor = gray]
    }
    
    {rank = same; mu; lambda}
    
    {m, beta} -> mu -> x;
    {nu, w} -> lambda -> x;
    mu -> lambda [dir = back]; ## (エッジ設定は、ノード位置の共通化用の小細工)
  }
")

# グラフを書出
DiagrammeRsvg::export_svg(gv = graph) |> # svgファイルに変換
  charToRaw() |> 
  rsvg::rsvg(height = 900) |> # ビットマップに変換
  png::writePNG(target = "figure/graphical_model/multivariate_gaussian_model_mean_precision.png", dpi = 100) # pngファイルに変換


