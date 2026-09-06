#import "@preview/catppuccin:1.0.1": catppuccin, flavors
#import "@preview/polylux:0.4.0": *

#let flavor = flavors.latte
// or: #let flavor = flavors.mocha
#let palette = flavor.colors
#show: catppuccin.with(flavor)
#let accent1 = it => text(fill: palette.flamingo.rgb, it)
#let accent2 = it => text(fill: palette.mauve.rgb, it)

#set page(
  paper: "presentation-16-9",
  margin: 1.3cm,
  // header: align(top, toolbox.full-width-block(progress-bar)),
  // footer: align(bottom, toolbox.full-width-block(inset: 10pt, context {
  //   text(size: 15pt, align(right, counter(page).display("1", both: false)))
  // }))
)
#set text(size: 22pt, font: "Lato")

#let team(name, content, car, ..paths) = slide({
  place(bottom + right, rotate(0deg, image(width: 10em, car)))
  grid(
    columns: (4em, auto),
    column-gutter: 2em,
    stack(
      dir: ttb,
      // spacing: .4em,
      spacing: 1fr,
      [],
      ..paths.map(it => image(width: 4em,it)),
      []
    ),
    stack(
      dir: ttb,
      spacing: 2em,
      text(size: 1.5em, name),
      content
    )
  )
})

#slide[
  #place(center + horizon, text(size: 3em)[FormulANN Grand Prix])
]

#slide[
  = Problem

  #v(1em)

  $k$-nearest neighbor queries: \ given a dataset $S$ and a point $q$, find the $k$ points in $S$ closest to $q$

  #line(length: 100%)

  - $k=100$
  - 7 datasets
    - dimensions 384 - 3072
    - size 200k - 1.4M
  - 1000 queries each
  - query set
    - public for development
    - secret for the competition
]

#slide[
  = Challenge prizes

  #v(1em)

  // #set list(spacing: 1.5em)
  / *Sherlock Holmes*: \
    fastest getting average recall $gt.eq 0.95$
  / *Bianconiglio*: \ fastest getting average recall $gt.eq 0.8$
  / *Dory*: \ approach using the least amount of memory with recall $gt.eq 0.95$
  / *Marie Kondo*: \ fastest at building the index
  / *Paperone*: \ the one using the fewest distance computations

  #place(top + right, rect(inset: .5em, radius: .4em)[
    #set text(.8em)
    #set align(left)
    #stack(dir: ttb,
      spacing: .8em,
      [1st - 10pts],
      [2nd - 8pts],
      [3rd - 6pts],
      [4th - 4pts],
      [5th - 3pts],
      [6th - 2ts],
      [7th - 1pts],
    )
  ])
]


// Graph based


#team(
  [Neighbors grass],
  [
    - Build HNSW graph on exact vectors
    - Keep only RaBitQ quantized vectors for queries
    - Never compute a full distance    
    - CPU with AVX512
    - C++
  ],
  "carart/neighbors-grass.svg",
  "imgs/giordani-francesco-circle.png",
  "imgs/moschetti-dario-circle.png",
  "imgs/tarantelli-kristjan-circle.png"
)

#team(
  [Close Enough],
  [
    - Build HNSW graph on compressed RaBitQ vectors
    - Tunes `ef` at build time, based on calibration queries
    - Exact reranking
    - CPU with AVX-512 instructions
    - C++
  ],
  "carart/close-enough.svg",
  "imgs/garagnani-filippo-circle.png",
  "imgs/vendramini-alberto-circle.png",
  "imgs/simoncini-marco-circle.png"
)

#team(
  [Kinda-neighbors],
  [
    - Build HNSW graph on 8-bit quantized vectors
    - Never computes a full distance
    - Adds a `patience` parameter at the base HNSW layer
    - Rust
    - CPU
  ],
  "carart/kinda-neighbors.svg",
  "imgs/dario-alessandro-circle.png",
  "imgs/visona-francesco-circle.png",
  "imgs/moretti-simone-circle.png"
)


// Partition based/codes

#team(
  [Friendly Neighborhood Solvers],
  [
    - Hierarchical k-means + quantization
    - CPU for ruting down the tree of clustering
    - GPU for scoring vector codes
    - C++
  ],
  "carart/friendly-neighborhood-solvers.svg",
  "imgs/chimisso-riccardo-circle.png",
  "imgs/mingardi-federico-circle.png",
  "imgs/montagnani-giulia-circle.png",
  "imgs/gambirasio-noemi-circle.png"
)

#team(
  [Finding NNemo],
  [
    - IVF + SAQ quantization (SIGMOD26)
    - No exact reranking
    - Fucus on minimizing memory footprint
    - CPU
    - C++
  ],
  "carart/finding-nnemo.svg",
  "imgs/coviello-antonella-circle.png",
  "imgs/beraldo-giulia-circle.png",
  "imgs/marchesini-mattia-circle.png",
  "imgs/zanon-stefano-circle.png"
)

// Pick your own

#team(
  [ANNarchy],
  [
    - HNSW or IVF, depending on the dataset
    - No quantization
    - CPU
    - C++
  ],
  "carart/annarchy.svg",
  "imgs/cappellato-ludovico-circle.png",
  "imgs/kalaj-nancy-circle.png",
  "imgs/mondin-silvia-circle.png"
)

#team(
  [ANNVedi],
  [
    - GPU: Brute force over 8-bit quantized representation
    - CPU: HNSW as a fallback for some datasets
    - Verify only a limited number of candidates
    - C++
  ],
  "carart/annvedi.svg",
  "imgs/nunziati-carlotta-circle.png",
  "imgs/canton-matteo-circle.png",
  "imgs/di-gennaro-veronica-circle.png"
)


#slide[
#set page(fill: rgb("#F2F1EC"))
  #set text(size: 2em)
  #set align(center + horizon)
  #link("http://localhost:8000/paddock.html", image(height: 100%, "circuits.png"))
]

#slide[
  = Conclusions
  
  - Quantization matters, a lot!
  - You can trust quantizer-estimated distances
  - GPU is viable even in a query-at-a-time scenario
]

