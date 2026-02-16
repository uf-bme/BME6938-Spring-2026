import { useCallback, useEffect, useState } from "react";
import 'katex/dist/katex.min.css'; // Don't forget the CSS

import { BlockMath, InlineMath } from "react-katex";

const tex = String.raw;

const slides = [
  // TITLE SLIDE
  {
    title: "Lecture: Classification Algorithms",
    content: (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-5xl font-bold text-center text-gray-800 mb-2">Classification Algorithms</div>
        <div className="text-2xl text-gray-500 mt-4 text-center">Logistic Regression, Random Forests, Gradient Boosting & MLPs</div>
        <div className="mt-12 flex gap-8">
          {["Logistic\nRegression","Random\nForests","Gradient\nBoosting","MLPs"].map((t,i)=>(
            <div key={i} className="w-36 h-28 rounded-xl flex items-center justify-center text-white font-bold text-center text-sm shadow-lg"
              style={{background:["#e74c3c","#27ae60","#f39c12","#8e44ad"][i]}}>
              {t.split("\n").map((l,j)=><div key={j}>{l}</div>)}
            </div>
          ))}
        </div>
        <div className="mt-10 text-gray-400 text-sm">BME 6938 Medical Artificial Intelligence</div>
      </div>
    )
  },
  // OVERVIEW
  {
    title: "Overview",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-8 text-gray-800">Lecture Overview</h2>
        <div className="grid grid-cols-2 gap-6">
          {[
            {n:"1",t:"Logistic Regression",d:"Linear decision boundary, sigmoid function, probabilistic output",c:"#e74c3c"},
            {n:"2",t:"Random Forests",d:"Ensemble of decision trees, bagging, feature importance",c:"#27ae60"},
            {n:"3",t:"Gradient Boosting",d:"Sequential ensemble, additive modeling, XGBoost",c:"#f39c12"},
            {n:"4",t:"Multilayer Perceptrons",d:"Neural networks, backpropagation, activation functions",c:"#8e44ad"}
          ].map((s,i)=>(
            <div key={i} className="border-2 rounded-xl p-5 flex items-start gap-4" style={{borderColor:s.c}}>
              <div className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0" style={{background:s.c}}>{s.n}</div>
              <div>
                <div className="font-bold text-lg" style={{color:s.c}}>{s.t}</div>
                <div className="text-gray-600 text-sm mt-1">{s.d}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  },
  // SECTION 1: LOGISTIC REGRESSION
  {
    title: "1 — Logistic Regression",
    content: (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-5xl font-bold text-white">1</div>
          <div className="text-4xl font-bold text-white mt-4">Logistic Regression</div>
        </div>
      </div>
    ),
    bg: "#e74c3c"
  },
  // LR: Intuition
  {
    title: "Logistic Regression — Intuition",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Logistic Regression — Intuition</h2>
        <div className="flex gap-8">
          <div className="flex-1">
            <p className="text-gray-700 mb-4 leading-relaxed">
              Despite its name, logistic regression is a <strong>classification</strong> algorithm. It models the probability that an input belongs to a particular class.
            </p>
            <p className="text-gray-700 mb-4 leading-relaxed">
              <strong>Key idea:</strong> Take a linear combination of features and pass it through a <span className="text-red-600 font-semibold">sigmoid function</span> to squash the output between 0 and 1.
            </p>
            <div className="bg-gray-100 rounded-xl p-4 mt-4">
              <div className="text-center text-lg">
                <InlineMath math={tex`z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b`} />
              </div>
              <div className="text-center text-lg mt-2">
                <InlineMath math={tex`\sigma(z) = \frac{1}{1 + e^{-z}}`} />
              </div>
              <div className="text-center text-lg mt-2">
                <InlineMath math={tex`\hat{y} = \sigma(z) \in [0,1]`} />
              </div>
            </div>
            <p className="text-gray-600 text-sm mt-4">
              If <InlineMath math={"\\sigma(z) \\ge 0.5"} /> → predict class 1, else class 0
            </p>
          </div>
          <div className="flex-1 flex flex-col items-center justify-center">
            <svg viewBox="0 0 300 200" className="w-full max-w-xs">
              <line x1="30" y1="170" x2="270" y2="170" stroke="#333" strokeWidth="2"/>
              <line x1="30" y1="170" x2="30" y2="20" stroke="#333" strokeWidth="2"/>
              <text x="150" y="195" textAnchor="middle" fontSize="12" fill="#666">z</text>
              <text x="15" y="95" textAnchor="middle" fontSize="12" fill="#666" transform="rotate(-90,15,95)">σ(z)</text>
              <text x="35" y="30" fontSize="10" fill="#999">1.0</text>
              <text x="35" y="100" fontSize="10" fill="#999">0.5</text>
              <text x="35" y="168" fontSize="10" fill="#999">0.0</text>
              <line x1="30" y1="30" x2="270" y2="30" stroke="#ddd" strokeDasharray="4"/>
              <line x1="30" y1="100" x2="270" y2="100" stroke="#ddd" strokeDasharray="4"/>
              <path d="M30,165 C60,164 90,160 120,145 C135,135 145,120 150,100 C155,80 165,65 180,55 C210,40 240,36 270,35" fill="none" stroke="#e74c3c" strokeWidth="3"/>
              <text x="200" y="75" fontSize="11" fill="#e74c3c" fontWeight="bold">Sigmoid</text>
            </svg>
            <div className="text-sm text-gray-500 mt-2">The sigmoid (logistic) function</div>
          </div>
        </div>
      </div>
    )
  },
  // LR: Decision Boundary
  {
    title: "Logistic Regression — Decision Boundary",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Decision Boundary</h2>
        <div className="flex gap-8">
          <div className="flex-1">
            <p className="text-gray-700 mb-4 leading-relaxed">Logistic regression produces a <strong>linear decision boundary</strong> in feature space.</p>
            <p className="text-gray-700 mb-4 leading-relaxed">
              The boundary is where <InlineMath math={"\\sigma(z)=0.5"} />, which means <InlineMath math={"z=0"} />:
            </p>
            <div className="bg-gray-100 rounded-xl p-4 font-mono text-center">
              <InlineMath math={"w_1x_1 + w_2x_2 + b = 0"} />
            </div>
            <div className="mt-6">
              <div className="font-semibold text-gray-700 mb-2">Strengths:</div>
              <p className="text-gray-600 text-sm mb-1">• Interpretable — coefficients show feature importance</p>
              <p className="text-gray-600 text-sm mb-1">• Outputs calibrated probabilities</p>
              <p className="text-gray-600 text-sm mb-1">• Fast to train, works well on linearly separable data</p>
            </div>
            <div className="mt-4">
              <div className="font-semibold text-gray-700 mb-2">Limitations:</div>
              <p className="text-gray-600 text-sm mb-1">• Cannot capture non-linear relationships (without feature engineering)</p>
              <p className="text-gray-600 text-sm">• Assumes features are roughly independent</p>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center">
            <svg viewBox="0 0 280 260" className="w-full max-w-xs">
              <rect x="30" y="10" width="240" height="240" fill="#f9f9f9" stroke="#ccc"/>
              <text x="150" y="265" textAnchor="middle" fontSize="12" fill="#666">x₁ (Feature 1)</text>
              <text x="12" y="130" textAnchor="middle" fontSize="12" fill="#666" transform="rotate(-90,12,130)">x₂ (Feature 2)</text>
              <line x1="30" y1="240" x2="260" y2="20" stroke="#e74c3c" strokeWidth="2.5" strokeDasharray="6"/>
              {[[60,200],[80,180],[70,160],[100,190],[120,210],[90,220],[110,170],[55,170],[130,180],[75,210]].map(([x,y],i)=>(
                <circle key={`b${i}`} cx={x} cy={y} r="6" fill="#3498db" opacity="0.7"/>
              ))}
              {[[180,60],[200,40],[170,80],[220,50],[190,90],[210,70],[240,30],[160,60],[230,80],[200,100]].map(([x,y],i)=>(
                <circle key={`r${i}`} cx={x} cy={y} r="6" fill="#e67e22" opacity="0.7"/>
              ))}
              <text x="70" y="145" fontSize="11" fill="#3498db" fontWeight="bold">Class 0</text>
              <text x="195" y="120" fontSize="11" fill="#e67e22" fontWeight="bold">Class 1</text>
              <text x="175" y="180" fontSize="10" fill="#e74c3c">Decision</text>
              <text x="175" y="192" fontSize="10" fill="#e74c3c">Boundary</text>
            </svg>
          </div>
        </div>
      </div>
    )
  },
  // LR: Loss Function
  {
    title: "Logistic Regression — Loss Function",
    content: (
      <div className="p-6">
        <h2 className="text-3xl font-bold mb-4 text-gray-800">Loss Function: Binary Cross-Entropy</h2>

        <div className="grid grid-cols-2 gap-6 items-start">
          {/* Left: formula + quick interpretation */}
          <div>
            <p className="text-gray-700 mb-3 text-sm leading-relaxed">
              Logistic regression is trained by minimizing the <strong>binary cross-entropy</strong> (log loss):
            </p>

            <div className="bg-gray-100 rounded-xl p-4 text-center mb-4">
              <BlockMath
                math={
                  "L = -\\frac{1}{n}\\sum_{i=1}^{n}\\left[y_i\\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)\\right]"
                }
              />
            </div>

            <div className="space-y-3">
              <div className="bg-blue-50 rounded-xl p-4">
                <div className="font-bold text-blue-700 mb-1 text-sm">If y = 1</div>
                <div className="text-gray-700 text-sm">
                  <InlineMath math={"\\ell(\\hat{y},1) = -\\log(\\hat{y})"} />
                </div>
                <div className="text-gray-600 text-xs mt-1">Confident & correct (ŷ → 1) ⇒ loss → 0</div>
              </div>

              <div className="bg-orange-50 rounded-xl p-4">
                <div className="font-bold text-orange-700 mb-1 text-sm">If y = 0</div>
                <div className="text-gray-700 text-sm">
                  <InlineMath math={"\\ell(\\hat{y},0) = -\\log(1-\\hat{y})"} />
                </div>
                <div className="text-gray-600 text-xs mt-1">Confident & correct (ŷ → 0) ⇒ loss → 0</div>
              </div>
            </div>

            <p className="text-gray-600 text-xs mt-4">
              Optimization: typically via <strong>gradient descent</strong> or second-order solvers (L-BFGS, Newton).
            </p>
          </div>

          {/* Right: graph */}
          <div className="bg-white border rounded-xl p-4">
            <div className="font-bold text-gray-800 mb-2 text-sm">Binary Cross-Entropy vs. Predicted Probability</div>

            <svg viewBox="0 0 320 220" className="w-full">
              {(() => {
                const x0 = 40;
                const y0 = 190;
                const x1 = 300;
                const y1 = 30;
                const w = x1 - x0;
                const h = y0 - y1;

                // Max loss for plotting (approx -log(0.01))
                const maxLoss = 4.6;

                const px = (p: number) => x0 + p * w;
                const py = (loss: number) => y0 - (Math.min(loss, maxLoss) / maxLoss) * h;

                const y1Points = Array.from({ length: 99 }, (_, i) => {
                  const p = (i + 1) / 100; // 0.01..0.99
                  const loss = -Math.log(p);
                  return `${px(p).toFixed(2)},${py(loss).toFixed(2)}`;
                }).join(" ");

                const y0Points = Array.from({ length: 99 }, (_, i) => {
                  const p = (i + 1) / 100; // 0.01..0.99
                  const loss = -Math.log(1 - p);
                  return `${px(p).toFixed(2)},${py(loss).toFixed(2)}`;
                }).join(" ");

                return (
                  <>
                    {/* Axes */}
                    <line x1={x0} y1={y0} x2={x1} y2={y0} stroke="#333" strokeWidth="1.5" />
                    <line x1={x0} y1={y0} x2={x0} y2={y1} stroke="#333" strokeWidth="1.5" />

                    {/* Gridlines */}
                    {[0, 1, 2, 3, 4].map((v) => {
                      const yy = py(v);
                      return (
                        <g key={v}>
                          <line x1={x0} y1={yy} x2={x1} y2={yy} stroke="#e5e7eb" strokeWidth="1" />
                          <text x={x0 - 8} y={yy + 4} textAnchor="end" fontSize="10" fill="#6b7280">
                            {v}
                          </text>
                        </g>
                      );
                    })}

                    {/* Curves */}
                    <polyline points={y1Points} fill="none" stroke="#2563eb" strokeWidth="2.5" />
                    <polyline points={y0Points} fill="none" stroke="#f97316" strokeWidth="2.5" />

                    {/* Labels */}
                    <text x={(x0 + x1) / 2} y={214} textAnchor="middle" fontSize="11" fill="#6b7280">
                      predicted probability ŷ
                    </text>
                    <text
                      x={14}
                      y={(y0 + y1) / 2}
                      textAnchor="middle"
                      fontSize="11"
                      fill="#6b7280"
                      transform={`rotate(-90,14,${(y0 + y1) / 2})`}
                    >
                      loss
                    </text>

                    {/* Ticks */}
                    {[0, 0.5, 1].map((p) => (
                      <g key={p}>
                        <line x1={px(p)} y1={y0} x2={px(p)} y2={y0 + 5} stroke="#333" strokeWidth="1" />
                        <text x={px(p)} y={y0 + 18} textAnchor="middle" fontSize="10" fill="#6b7280">
                          {p}
                        </text>
                      </g>
                    ))}

                    {/* Legend */}
                    <rect x="170" y="38" width="130" height="44" rx="8" fill="white" stroke="#e5e7eb" />
                    <line x1="182" y1="54" x2="202" y2="54" stroke="#2563eb" strokeWidth="2.5" />
                    <text x="208" y="57" fontSize="10" fill="#374151">
                      y = 1 : -log(ŷ)
                    </text>
                    <line x1="182" y1="72" x2="202" y2="72" stroke="#f97316" strokeWidth="2.5" />
                    <text x="208" y="75" fontSize="10" fill="#374151">
                      y = 0 : -log(1-ŷ)
                    </text>
                  </>
                );
              })()}
            </svg>

            <div className="text-xs text-gray-500 mt-2">
              Wrong confident predictions are penalized heavily (loss increases steeply near 0 or 1).
            </div>
          </div>
        </div>
      </div>
    )
  },
  // LR: Medical Example
  {
    title: "Logistic Regression — Medical Example",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Medical Application Example</h2>
        <div className="bg-red-50 border-2 border-red-200 rounded-xl p-6 mb-6">
          <div className="font-bold text-red-700 text-lg mb-2">Predicting Diabetes Risk</div>
          <p className="text-gray-700">Given patient features (glucose level, BMI, age, blood pressure), predict probability of diabetes (0 or 1).</p>
        </div>
        <div className="bg-gray-100 rounded-xl p-5 mb-4">
          <div className="text-sm text-center">
            <InlineMath
              math={
                "P(\\mathrm{diabetes}=1) = \\sigma(0.03\\cdot\\mathrm{glucose} + 0.05\\cdot\\mathrm{BMI} + 0.01\\cdot\\mathrm{age} - 4.2)"
              }
            />
          </div>
        </div>
        <div className="flex gap-6">
          <div className="flex-1 bg-white border rounded-lg p-4">
            <div className="font-bold text-gray-700 mb-2">Interpretation</div>
            <p className="text-sm text-gray-600">Each coefficient tells us the <strong>log-odds change</strong> per unit increase. E.g., each 1-unit increase in BMI increases the log-odds by 0.05.</p>
          </div>
          <div className="flex-1 bg-white border rounded-lg p-4">
            <div className="font-bold text-gray-700 mb-2">Why use it in medicine?</div>
            <p className="text-sm text-gray-600">Clinicians can understand and trust the model — it's transparent, outputs probabilities, and coefficients map to clinical risk factors.</p>
          </div>
        </div>
      </div>
    )
  },
  // SECTION 2: RANDOM FORESTS
  {
    title: "2 — Random Forests",
    content: (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-5xl font-bold text-white">2</div>
          <div className="text-4xl font-bold text-white mt-4">Random Forests</div>
        </div>
      </div>
    ),
    bg: "#27ae60"
  },
  // RF: Decision Trees Recap
  {
    title: "Random Forests — Decision Trees Recap",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">First: Decision Trees</h2>
        <div className="flex gap-8">
          <div className="flex-1">
            <p className="text-gray-700 mb-4 leading-relaxed">A decision tree recursively splits data based on feature thresholds to create a tree-like model of decisions.</p>
            <p className="text-gray-700 mb-2 font-semibold">Splitting criteria:</p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-3">
              <p className="text-sm font-semibold text-blue-700 mb-1">Gini Impurity:</p>
              <div className="text-center mb-1">
                <InlineMath math={"\\mathrm{Gini}(S) = 1 - \\sum_{i=1}^{c} p_i^2"} />
              </div>
              <p className="text-xs text-gray-600">p_i = proportion of class i. Gini = 0 when pure, Gini = 0.5 for balanced binary split.</p>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-3">
              <p className="text-sm font-semibold text-green-700 mb-1">Entropy / Information Gain:</p>
              <div className="text-center mb-1">
                <InlineMath math={"H(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)"} />
              </div>
              <p className="text-xs text-gray-600">Higher entropy = more disorder. Information gain = reduction in entropy after split.</p>
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-sm text-gray-700"><strong>Problem:</strong> A single decision tree is prone to <span className="text-red-600 font-semibold">overfitting</span> — it memorizes noise in the training data.</p>
              <p className="text-sm text-gray-700 mt-2"><strong>Solution:</strong> Combine many trees → <span className="text-green-600 font-semibold">Random Forest</span></p>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center">
            <svg viewBox="0 0 300 240" className="w-full max-w-sm">
              <rect x="100" y="10" width="100" height="35" rx="5" fill="#27ae60" stroke="#1e8449" strokeWidth="1.5"/>
              <text x="150" y="32" textAnchor="middle" fontSize="10" fill="white" fontWeight="bold">Glucose ≥ 140?</text>
              <line x1="130" y1="45" x2="75" y2="80" stroke="#555" strokeWidth="1.5"/>
              <line x1="170" y1="45" x2="225" y2="80" stroke="#555" strokeWidth="1.5"/>
              <text x="95" y="65" fontSize="9" fill="#27ae60">Yes</text>
              <text x="190" y="65" fontSize="9" fill="#e74c3c">No</text>
              <rect x="25" y="80" width="100" height="35" rx="5" fill="#27ae60" stroke="#1e8449" strokeWidth="1.5"/>
              <text x="75" y="102" textAnchor="middle" fontSize="10" fill="white" fontWeight="bold">BMI ≥ 30?</text>
              <rect x="175" y="80" width="100" height="35" rx="5" fill="#3498db" stroke="#2980b9" strokeWidth="1.5"/>
              <text x="225" y="102" textAnchor="middle" fontSize="10" fill="white" fontWeight="bold">Age ≥ 50?</text>
              <line x1="55" y1="115" x2="30" y2="150" stroke="#555" strokeWidth="1.5"/>
              <line x1="95" y1="115" x2="120" y2="150" stroke="#555" strokeWidth="1.5"/>
              <line x1="205" y1="115" x2="185" y2="150" stroke="#555" strokeWidth="1.5"/>
              <line x1="245" y1="115" x2="260" y2="150" stroke="#555" strokeWidth="1.5"/>
              <rect x="5" y="150" width="55" height="30" rx="5" fill="#e74c3c"/>
              <text x="32" y="169" textAnchor="middle" fontSize="9" fill="white" fontWeight="bold">Diabetic</text>
              <rect x="95" y="150" width="55" height="30" rx="5" fill="#2ecc71"/>
              <text x="122" y="169" textAnchor="middle" fontSize="9" fill="white" fontWeight="bold">Healthy</text>
              <rect x="160" y="150" width="55" height="30" rx="5" fill="#e74c3c"/>
              <text x="187" y="169" textAnchor="middle" fontSize="9" fill="white" fontWeight="bold">Diabetic</text>
              <rect x="237" y="150" width="55" height="30" rx="5" fill="#2ecc71"/>
              <text x="264" y="169" textAnchor="middle" fontSize="9" fill="white" fontWeight="bold">Healthy</text>
              <text x="150" y="210" textAnchor="middle" fontSize="11" fill="#888">Example Decision Tree</text>
            </svg>
          </div>
        </div>
      </div>
    )
  },
  // RF: How it works
  {
    title: "Random Forests — How It Works",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Random Forest — Ensemble of Trees</h2>
        <div className="flex gap-6 mb-6">
          <div className="flex-1">
            <p className="text-gray-700 mb-4 leading-relaxed">A Random Forest builds <strong>many decision trees</strong> and combines their predictions through <strong>majority voting</strong> (classification) or <strong>averaging</strong> (regression).</p>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-green-600 text-white flex items-center justify-center font-bold text-sm flex-shrink-0">1</div>
                <div><strong className="text-green-700">Bagging:</strong> <span className="text-gray-600 text-sm">Each tree is trained on a random bootstrap sample (sampling with replacement) of the data.</span></div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-green-600 text-white flex items-center justify-center font-bold text-sm flex-shrink-0">2</div>
                <div><strong className="text-green-700">Feature randomness:</strong> <span className="text-gray-600 text-sm">At each split, only a random subset of features (√p for classification) is considered.</span></div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-green-600 text-white flex items-center justify-center font-bold text-sm flex-shrink-0">3</div>
                <div><strong className="text-green-700">Aggregate:</strong> <span className="text-gray-600 text-sm">Final prediction = majority vote across all trees.</span></div>
              </div>
            </div>
          </div>
          <div className="flex-1 flex flex-col items-center justify-center">
            <div className="flex gap-4 mb-4">
              {[1,2,3,4,5].map(i=>(
                <div key={i} className="flex flex-col items-center">
                  <div className="text-2xl">🌲</div>
                  <div className="text-xs text-gray-500">Tree {i}</div>
                </div>
              ))}
            </div>
            <div className="text-gray-400 text-2xl mb-2">↓ ↓ ↓ ↓ ↓</div>
            <div className="bg-green-100 border-2 border-green-500 rounded-xl px-6 py-3 font-bold text-green-700">Majority Vote → Final Prediction</div>
          </div>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-sm text-gray-700"><strong>Why does this work?</strong> Each tree overfits differently. By averaging many diverse trees, the individual errors cancel out → lower variance without increasing bias much.</p>
        </div>
      </div>
    )
  },
  // RF: Bagging visual
  {
    title: "Random Forests — Bagging",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Bootstrap Aggregation (Bagging)</h2>
        <div className="flex flex-col items-center">
          <div className="bg-gray-200 rounded-lg px-8 py-3 font-bold text-gray-700 mb-4">Original Dataset (N samples)</div>
          <div className="text-gray-400 text-xl mb-2">↓ Sample with replacement</div>
          <div className="flex gap-4 mb-4">
            {["Bootstrap 1","Bootstrap 2","Bootstrap 3","...","Bootstrap B"].map((l,i)=>(
              <div key={i} className="bg-green-100 border border-green-400 rounded-lg px-4 py-2 text-sm font-semibold text-green-700">{l}</div>
            ))}
          </div>
          <div className="text-gray-400 text-xl mb-2">↓ Train independently</div>
          <div className="flex gap-4 mb-4">
            {["Tree 1","Tree 2","Tree 3","...","Tree B"].map((l,i)=>(
              <div key={i} className="bg-green-600 text-white rounded-lg px-4 py-2 text-sm font-semibold">{l}</div>
            ))}
          </div>
          <div className="text-gray-400 text-xl mb-2">↓ Aggregate</div>
          <div className="bg-green-700 text-white rounded-xl px-8 py-3 font-bold text-lg">Final Prediction</div>
        </div>
        <div className="mt-6 grid grid-cols-2 gap-4">
          <div className="bg-gray-50 border rounded-lg p-3 text-sm text-gray-600">
            <strong>~63.2%</strong> of original samples appear in each bootstrap (on average). The remaining ~36.8% are "out-of-bag" (OOB) samples used for validation.
          </div>
          <div className="bg-gray-50 border rounded-lg p-3 text-sm text-gray-600">
            <strong>Key hyperparameters:</strong> number of trees (n_estimators), max depth, min samples per leaf, max features per split.
          </div>
        </div>
      </div>
    )
  },
  // RF: Pros/Cons
  {
    title: "Random Forests — Strengths & Limitations",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Random Forests — Strengths & Limitations</h2>
        <div className="grid grid-cols-2 gap-8">
          <div className="bg-green-50 border-2 border-green-300 rounded-xl p-6">
            <div className="text-xl font-bold text-green-700 mb-4">✅ Strengths</div>
            <div className="space-y-3 text-gray-700 text-sm">
              <p>• <strong>Robust to overfitting</strong> — averaging many trees reduces variance</p>
              <p>• <strong>Handles non-linear</strong> relationships naturally</p>
              <p>• <strong>No feature scaling</strong> required</p>
              <p>• <strong>Feature importance</strong> built-in (Gini importance, permutation importance)</p>
              <p>• <strong>Works well out-of-the-box</strong> with minimal tuning</p>
              <p>• <strong>Handles missing data</strong> and mixed feature types</p>
            </div>
          </div>
          <div className="bg-red-50 border-2 border-red-300 rounded-xl p-6">
            <div className="text-xl font-bold text-red-700 mb-4">❌ Limitations</div>
            <div className="space-y-3 text-gray-700 text-sm">
              <p>• <strong>Less interpretable</strong> than a single tree or logistic regression</p>
              <p>• <strong>Slower inference</strong> — must evaluate all trees</p>
              <p>• <strong>Memory intensive</strong> — stores many full trees</p>
              <p>• <strong>Can struggle</strong> with very high-dimensional sparse data</p>
              <p>• <strong>Biased toward</strong> features with many levels (Gini importance)</p>
              <p>• <strong>Cannot extrapolate</strong> beyond training data range</p>
            </div>
          </div>
        </div>
      </div>
    )
  },
  // SECTION 3: GRADIENT BOOSTING
  {
    title: "3 — Gradient Boosting",
    content: (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-5xl font-bold text-white">3</div>
          <div className="text-4xl font-bold text-white mt-4">Gradient Boosting</div>
        </div>
      </div>
    ),
    bg: "#f39c12"
  },
  // GB: Intuition
  {
    title: "Gradient Boosting — Intuition",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Gradient Boosting — Intuition</h2>
        <div className="flex gap-8">
          <div className="flex-1">
            <p className="text-gray-700 mb-4 leading-relaxed">
              While Random Forests train trees <strong>in parallel</strong> (independently), Gradient Boosting trains trees <strong>sequentially</strong> — each new tree corrects the errors of the previous ones.
            </p>
            <div className="bg-yellow-50 border border-yellow-300 rounded-xl p-4 mb-4">
              <p className="text-gray-700 font-semibold mb-2">Core idea:</p>
              <p className="text-gray-600 text-sm">1. Start with a simple prediction (e.g., mean)</p>
              <p className="text-gray-600 text-sm">2. Compute the <strong>residuals</strong> (errors)</p>
              <p className="text-gray-600 text-sm">3. Fit a new small tree to predict the residuals</p>
              <p className="text-gray-600 text-sm">4. Add this tree's predictions (scaled by learning rate) to the model</p>
              <p className="text-gray-600 text-sm">5. Repeat steps 2–4</p>
            </div>
            <div className="bg-gray-100 rounded-lg p-3 font-mono text-sm text-center">
              F(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">η = learning rate, hₘ = weak learner (small tree)</p>
          </div>
          <div className="flex-1 flex flex-col items-center justify-center gap-3">
            {["Model₀: Initial guess","+ Tree₁: Fix biggest errors","+ Tree₂: Fix remaining errors","+ Tree₃: Fine-tune","...","= Final Strong Model"].map((t,i)=>(
              <div key={i} className={`w-64 px-4 py-2 rounded-lg text-sm font-semibold text-center ${i===5?"bg-orange-500 text-white":"bg-orange-100 text-orange-800 border border-orange-300"}`}>{t}</div>
            ))}
          </div>
        </div>
      </div>
    )
  },
  // GB: Boosting vs Bagging
  {
    title: "Gradient Boosting — Boosting vs Bagging",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Boosting vs. Bagging</h2>
        <div className="grid grid-cols-2 gap-8">
          <div className="border-2 border-green-500 rounded-xl p-6">
            <div className="text-xl font-bold text-green-600 mb-4">Bagging (Random Forest)</div>
            <div className="flex flex-col items-center mb-4">
              <div className="flex gap-2">
                {[1,2,3,4].map(i=>(<div key={i} className="w-12 h-14 bg-green-200 rounded flex items-center justify-center text-xs font-bold text-green-700">T{i}</div>))}
              </div>
              <div className="text-sm text-gray-500 mt-1">Independent & Parallel</div>
            </div>
            <p className="text-sm text-gray-600 mb-1">• Trees trained independently</p>
            <p className="text-sm text-gray-600 mb-1">• Each on a bootstrap sample</p>
            <p className="text-sm text-gray-600 mb-1">• Reduces <strong>variance</strong></p>
            <p className="text-sm text-gray-600">• Deep trees (low bias, high variance)</p>
          </div>
          <div className="border-2 border-orange-500 rounded-xl p-6">
            <div className="text-xl font-bold text-orange-600 mb-4">Boosting (Gradient Boosting)</div>
            <div className="flex flex-col items-center mb-4">
              <div className="flex gap-1 items-center">
                {[1,2,3,4].map(i=>(<>
                  <div key={i} className="w-12 h-14 bg-orange-200 rounded flex items-center justify-center text-xs font-bold text-orange-700">T{i}</div>
                  {i<4 && <div className="text-orange-400 font-bold">→</div>}
                </>))}
              </div>
              <div className="text-sm text-gray-500 mt-1">Sequential & Dependent</div>
            </div>
            <p className="text-sm text-gray-600 mb-1">• Trees trained sequentially</p>
            <p className="text-sm text-gray-600 mb-1">• Each corrects previous errors</p>
            <p className="text-sm text-gray-600 mb-1">• Reduces <strong>bias</strong></p>
            <p className="text-sm text-gray-600">• Shallow trees (high bias, low variance)</p>
          </div>
        </div>
        <div className="mt-6 bg-gray-100 rounded-lg p-4 text-sm text-gray-700 text-center">
          Think of it this way: <strong>Bagging</strong> = many experts vote independently. <strong>Boosting</strong> = each new expert focuses on what previous experts got wrong.
        </div>
      </div>
    )
  },
  // GB: Popular Implementations
  {
    title: "Gradient Boosting — XGBoost, LightGBM, CatBoost",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Popular Implementations</h2>
        <div className="grid grid-cols-3 gap-6 mb-6">
          {[
            {name:"XGBoost",desc:"Extreme Gradient Boosting. Regularized objective, efficient handling of sparse data, parallelized tree construction.",color:"#e74c3c",year:"2014"},
            {name:"LightGBM",desc:"Light Gradient Boosting Machine. Leaf-wise growth, histogram-based splitting, very fast on large datasets.",color:"#f39c12",year:"2017"},
            {name:"CatBoost",desc:"Categorical Boosting. Native handling of categorical features, ordered boosting to reduce overfitting.",color:"#3498db",year:"2017"}
          ].map((lib,i)=>(
            <div key={i} className="border-2 rounded-xl p-5" style={{borderColor:lib.color}}>
              <div className="font-bold text-lg mb-1" style={{color:lib.color}}>{lib.name}</div>
              <div className="text-xs text-gray-400 mb-2">{lib.year}</div>
              <p className="text-sm text-gray-600">{lib.desc}</p>
            </div>
          ))}
        </div>
        <div className="bg-orange-50 border border-orange-200 rounded-xl p-5">
          <div className="font-bold text-orange-700 mb-2">Key Hyperparameters</div>
          <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
            <p>• <strong>n_estimators:</strong> number of boosting rounds</p>
            <p>• <strong>learning_rate (η):</strong> shrinkage per step (0.01–0.3)</p>
            <p>• <strong>max_depth:</strong> depth of each tree (3–8 typical)</p>
            <p>• <strong>subsample:</strong> fraction of data per tree</p>
            <p>• <strong>colsample_bytree:</strong> fraction of features per tree</p>
            <p>• <strong>reg_alpha / reg_lambda:</strong> L1/L2 regularization</p>
          </div>
        </div>
      </div>
    )
  },
  // GB: Overfitting Warning
  {
    title: "Gradient Boosting — Overfitting",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Overfitting in Gradient Boosting</h2>
        <p className="text-gray-700 mb-6">Gradient boosting is more prone to overfitting than Random Forests. The learning rate and number of trees must be carefully tuned.</p>
        <div className="flex gap-8 items-center">
          <div className="flex-1">
            <svg viewBox="0 0 320 220" className="w-full">
              <line x1="40" y1="190" x2="300" y2="190" stroke="#333" strokeWidth="1.5"/>
              <line x1="40" y1="190" x2="40" y2="20" stroke="#333" strokeWidth="1.5"/>
              <text x="170" y="215" textAnchor="middle" fontSize="11" fill="#666">Number of Trees (Boosting Rounds)</text>
              <text x="15" y="105" textAnchor="middle" fontSize="11" fill="#666" transform="rotate(-90,15,105)">Error</text>
              <path d="M50,40 C80,50 120,80 160,110 C200,135 240,155 290,170" fill="none" stroke="#3498db" strokeWidth="2.5"/>
              <text x="260" y="162" fontSize="10" fill="#3498db" fontWeight="bold">Training</text>
              <path d="M50,60 C80,68 120,85 160,95 C180,100 200,102 220,108 C250,120 270,140 290,160" fill="none" stroke="#e74c3c" strokeWidth="2.5"/>
              <text x="260" y="148" fontSize="10" fill="#e74c3c" fontWeight="bold">Validation</text>
              <line x1="200" y1="25" x2="200" y2="185" stroke="#f39c12" strokeWidth="1.5" strokeDasharray="5"/>
              <text x="200" y="18" textAnchor="middle" fontSize="10" fill="#f39c12" fontWeight="bold">Optimal</text>
              <text x="245" y="40" fontSize="9" fill="#999">Overfitting</text>
              <text x="245" y="50" fontSize="9" fill="#999">zone →</text>
            </svg>
          </div>
          <div className="flex-1 space-y-4">
            <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4">
              <div className="font-bold text-yellow-700 mb-1">Prevention strategies:</div>
              <p className="text-sm text-gray-600">• Use <strong>early stopping</strong> on validation loss</p>
              <p className="text-sm text-gray-600">• Lower the <strong>learning rate</strong> (with more trees)</p>
              <p className="text-sm text-gray-600">• Limit <strong>tree depth</strong> (3–6)</p>
              <p className="text-sm text-gray-600">• Add <strong>regularization</strong> (L1/L2)</p>
              <p className="text-sm text-gray-600">• Use <strong>subsampling</strong> (stochastic GB)</p>
            </div>
          </div>
        </div>
      </div>
    )
  },
  // SECTION 4: MLPs
  {
    title: "4 — Multilayer Perceptrons",
    content: (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-5xl font-bold text-white">4</div>
          <div className="text-4xl font-bold text-white mt-4">Multilayer Perceptrons (MLPs)</div>
        </div>
      </div>
    ),
    bg: "#8e44ad"
  },
  // MLP: Single Neuron
  {
    title: "MLPs — The Perceptron",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Building Block: The Perceptron</h2>
        <div className="flex gap-8">
          <div className="flex-1">
            <p className="text-gray-700 mb-4 leading-relaxed">A single perceptron is essentially logistic regression — it computes a weighted sum and applies an activation function.</p>
            <div className="bg-gray-100 rounded-xl p-4 mb-4">
              <div className="text-center">
                <InlineMath math={"z = \\sum_i w_ix_i + b"} />
              </div>
              <div className="text-center mt-1">
                <InlineMath math={"a = f(z)"} />
              </div>
            </div>
            <p className="text-gray-700 mb-2 font-semibold">Common activation functions:</p>
            <p className="text-gray-600 text-sm mb-1">
              • <strong>Sigmoid:</strong> <InlineMath math={"\\sigma(z)=\\frac{1}{1+e^{-z}}"} /> — output [0,1]
            </p>
            <p className="text-gray-600 text-sm mb-1">
              • <strong>ReLU:</strong> <InlineMath math={"f(z)=\\max(0,z)"} /> — most popular for hidden layers
            </p>
            <p className="text-gray-600 text-sm mb-1">
              • <strong>Tanh:</strong> <InlineMath math={"f(z)=\\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}"} /> — output [−1,1]
            </p>
            <p className="text-gray-600 text-sm">• <strong>Softmax:</strong> for multi-class output layer</p>
          </div>
          <div className="flex-1 flex items-center justify-center">
            <svg viewBox="0 0 300 200" className="w-full max-w-sm">
              {[40,80,120,160].map((y,i)=>(
                <g key={i}>
                  <circle cx="40" cy={y} r="18" fill="#e8daef" stroke="#8e44ad" strokeWidth="1.5"/>
                  <text x="40" y={y+4} textAnchor="middle" fontSize="10" fill="#8e44ad" fontWeight="bold">x{i+1}</text>
                  <line x1="58" y1={y} x2="140" y2="100" stroke="#8e44ad" strokeWidth="1.5"/>
                  <text x="95" y={y<100?y+15:y-5} fontSize="8" fill="#999">w{i+1}</text>
                </g>
              ))}
              <circle cx="160" cy="100" r="25" fill="#8e44ad" stroke="#6c3483" strokeWidth="2"/>
              <text x="160" y="97" textAnchor="middle" fontSize="9" fill="white" fontWeight="bold">Σ + f</text>
              <text x="160" y="108" textAnchor="middle" fontSize="8" fill="#d5b8e8">(activation)</text>
              <line x1="185" y1="100" x2="260" y2="100" stroke="#8e44ad" strokeWidth="2" markerEnd="url(#arrow)"/>
              <circle cx="275" cy="100" r="15" fill="#f5eef8" stroke="#8e44ad" strokeWidth="1.5"/>
              <text x="275" y="104" textAnchor="middle" fontSize="10" fill="#8e44ad" fontWeight="bold">ŷ</text>
              <defs><marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8e44ad"/></marker></defs>
            </svg>
          </div>
        </div>
      </div>
    )
  },
  // MLP: Architecture
  {
    title: "MLPs — Network Architecture",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Multilayer Perceptron Architecture</h2>
        <p className="text-gray-700 mb-4">An MLP stacks multiple layers of neurons. Information flows forward from input → hidden layers → output (feedforward network).</p>
        <div className="flex items-center justify-center mb-6">
          <svg viewBox="0 0 450 220" className="w-full max-w-lg">
            {/* Input layer */}
            {[40,80,120,160].map((y,i)=>(
              <g key={`i${i}`}>
                <circle cx="50" cy={y} r="16" fill="#ebf5fb" stroke="#3498db" strokeWidth="2"/>
                <text x="50" y={y+4} textAnchor="middle" fontSize="9" fill="#3498db" fontWeight="bold">x{i+1}</text>
              </g>
            ))}
            <text x="50" y="195" textAnchor="middle" fontSize="10" fill="#3498db" fontWeight="bold">Input</text>
            {/* Hidden 1 */}
            {[30,70,110,150,190].map((y,i)=>(
              <g key={`h1${i}`}>
                <circle cx="170" cy={y} r="16" fill="#f5eef8" stroke="#8e44ad" strokeWidth="2"/>
                {[40,80,120,160].map((iy,j)=>(
                  <line key={j} x1="66" y1={iy} x2="154" y2={y} stroke="#d5d8dc" strokeWidth="0.7"/>
                ))}
              </g>
            ))}
            <text x="170" y="215" textAnchor="middle" fontSize="10" fill="#8e44ad" fontWeight="bold">Hidden 1</text>
            {/* Hidden 2 */}
            {[50,100,150].map((y,i)=>(
              <g key={`h2${i}`}>
                <circle cx="290" cy={y} r="16" fill="#f5eef8" stroke="#8e44ad" strokeWidth="2"/>
                {[30,70,110,150,190].map((hy,j)=>(
                  <line key={j} x1="186" y1={hy} x2="274" y2={y} stroke="#d5d8dc" strokeWidth="0.7"/>
                ))}
              </g>
            ))}
            <text x="290" y="215" textAnchor="middle" fontSize="10" fill="#8e44ad" fontWeight="bold">Hidden 2</text>
            {/* Output */}
            {[80,120].map((y,i)=>(
              <g key={`o${i}`}>
                <circle cx="400" cy={y} r="16" fill="#fdedec" stroke="#e74c3c" strokeWidth="2"/>
                <text x="400" y={y+4} textAnchor="middle" fontSize="9" fill="#e74c3c" fontWeight="bold">ŷ{i+1}</text>
                {[50,100,150].map((hy,j)=>(
                  <line key={j} x1="306" y1={hy} x2="384" y2={y} stroke="#d5d8dc" strokeWidth="0.7"/>
                ))}
              </g>
            ))}
            <text x="400" y="215" textAnchor="middle" fontSize="10" fill="#e74c3c" fontWeight="bold">Output</text>
          </svg>
        </div>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="bg-blue-50 rounded-lg p-3 text-center"><strong className="text-blue-600">Input layer:</strong><br/>One neuron per feature</div>
          <div className="bg-purple-50 rounded-lg p-3 text-center"><strong className="text-purple-600">Hidden layers:</strong><br/>Learn representations (ReLU)</div>
          <div className="bg-red-50 rounded-lg p-3 text-center"><strong className="text-red-600">Output layer:</strong><br/>Sigmoid (binary) or Softmax (multi-class)</div>
        </div>
        <div className="mt-4 text-center">
          <a href="https://playground.tensorflow.org/#activation=linear&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.25736&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false" target="_blank" rel="noopener noreferrer" className="inline-block px-6 py-2 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition">
            Try it: TensorFlow Playground
          </a>
        </div>
      </div>
    )
  },
  // MLP: Backpropagation
  {
    title: "MLPs — Training with Backpropagation",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Training: Backpropagation</h2>
        <div className="flex gap-8">
          <div className="flex-1">
            <p className="text-gray-700 mb-4 leading-relaxed">MLPs are trained using <strong>backpropagation</strong> with <strong>gradient descent</strong>. The chain rule propagates error gradients backward through the network to update all weights.</p>
            <div className="space-y-3 mb-4">
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-purple-600 text-white flex items-center justify-center font-bold text-sm flex-shrink-0">1</div>
                <div className="text-sm text-gray-600"><strong>Forward pass:</strong> compute predictions</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-purple-600 text-white flex items-center justify-center font-bold text-sm flex-shrink-0">2</div>
                <div className="text-sm text-gray-600"><strong>Compute loss:</strong> compare prediction to true label</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-purple-600 text-white flex items-center justify-center font-bold text-sm flex-shrink-0">3</div>
                <div className="text-sm text-gray-600"><strong>Backward pass:</strong> compute ∂L/∂w for every weight using chain rule</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-purple-600 text-white flex items-center justify-center font-bold text-sm flex-shrink-0">4</div>
                <div className="text-sm text-gray-600">
                  <strong>Update weights:</strong> <InlineMath math={"w \\leftarrow w - \\eta \\cdot \\frac{\\partial L}{\\partial w}"} />
                </div>
              </div>
            </div>
            <div className="bg-gray-100 rounded-lg p-3 font-mono text-sm text-center">
              <InlineMath math={"w_{\\text{new}} = w_{\\text{old}} - \\eta \\cdot \\frac{\\partial \\mathrm{Loss}}{\\partial w}"} />
            </div>
          </div>
          <div className="flex-1">
            <div className="bg-purple-50 border border-purple-200 rounded-xl p-5 mb-4">
              <div className="font-bold text-purple-700 mb-2">Common Optimizers</div>
              <p className="text-sm text-gray-600 mb-1">• <strong>SGD:</strong> basic stochastic gradient descent</p>
              <p className="text-sm text-gray-600 mb-1">• <strong>SGD + Momentum:</strong> smooths updates</p>
              <p className="text-sm text-gray-600 mb-1">• <strong>Adam:</strong> adaptive learning rates (most popular)</p>
              <p className="text-sm text-gray-600">• <strong>AdamW:</strong> Adam with weight decay</p>
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-5">
              <div className="font-bold text-yellow-700 mb-2">Regularization Techniques</div>
              <p className="text-sm text-gray-600 mb-1">• <strong>Dropout:</strong> randomly zero out neurons during training</p>
              <p className="text-sm text-gray-600 mb-1">• <strong>Weight decay:</strong> L2 penalty on weights</p>
              <p className="text-sm text-gray-600 mb-1">• <strong>Batch normalization:</strong> normalize layer inputs</p>
              <p className="text-sm text-gray-600">• <strong>Early stopping:</strong> stop when val loss stops improving</p>
            </div>
          </div>
        </div>
      </div>
    )
  },
  // MLP: Pros/Cons
  {
    title: "MLPs — Strengths & Limitations",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">MLPs — Strengths & Limitations</h2>
        <div className="grid grid-cols-2 gap-8 mb-6">
          <div className="bg-purple-50 border-2 border-purple-300 rounded-xl p-6">
            <div className="text-xl font-bold text-purple-700 mb-4">✅ Strengths</div>
            <div className="space-y-2 text-gray-700 text-sm">
              <p>• <strong>Universal approximator</strong> — can learn any continuous function (given enough neurons)</p>
              <p>• <strong>Flexible architecture</strong> — can be adapted to many tasks</p>
              <p>• <strong>Learns features automatically</strong> — no manual feature engineering</p>
              <p>• <strong>Scales well</strong> with GPU acceleration</p>
              <p>• <strong>Foundation</strong> for deep learning (CNNs, RNNs, Transformers)</p>
            </div>
          </div>
          <div className="bg-red-50 border-2 border-red-300 rounded-xl p-6">
            <div className="text-xl font-bold text-red-700 mb-4">❌ Limitations</div>
            <div className="space-y-2 text-gray-700 text-sm">
              <p>• <strong>Black box</strong> — difficult to interpret</p>
              <p>• <strong>Requires lots of data</strong> to train well</p>
              <p>• <strong>Many hyperparameters</strong> (architecture, LR, batch size, etc.)</p>
              <p>• <strong>Sensitive to feature scaling</strong> — must normalize inputs</p>
              <p>• <strong>Computationally expensive</strong> to train</p>
              <p>• <strong>Prone to overfitting</strong> on small datasets</p>
            </div>
          </div>
        </div>
        <div className="bg-gray-100 rounded-xl p-4 text-sm text-gray-700 text-center">
          <strong>Rule of thumb:</strong> For tabular/structured medical data with limited samples, tree-based methods (RF, XGBoost) often outperform MLPs. MLPs shine with large datasets and when combined with specialized architectures (CNNs for images, RNNs for time series).
        </div>
      </div>
    )
  },
  // COMPARISON SLIDE
  {
    title: "Model Comparison",
    content: (
      <div className="p-6">
        <h2 className="text-3xl font-bold mb-4 text-gray-800">Algorithm Comparison</h2>
        <div className="overflow-hidden rounded-xl border border-gray-200">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-800 text-white">
                <th className="p-3 text-left">Property</th>
                <th className="p-3 text-center" style={{color:"#ff7675"}}>Logistic Reg.</th>
                <th className="p-3 text-center" style={{color:"#55efc4"}}>Random Forest</th>
                <th className="p-3 text-center" style={{color:"#ffeaa7"}}>Gradient Boost</th>
                <th className="p-3 text-center" style={{color:"#d5aaff"}}>MLP</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Decision Boundary","Linear","Non-linear","Non-linear","Non-linear"],
                ["Interpretability","High","Medium","Low–Medium","Low"],
                ["Training Speed","Very Fast","Fast","Medium","Slow"],
                ["Feature Scaling","Required","Not needed","Not needed","Required"],
                ["Handles Non-linearity","No*","Yes","Yes","Yes"],
                ["Risk of Overfitting","Low","Low","Medium–High","High"],
                ["Data Size Needed","Small OK","Medium","Medium","Large"],
                ["Probabilistic Output","Yes","Yes","Yes","Yes"],
              ].map((row,i)=>(
                <tr key={i} className={i%2===0?"bg-white":"bg-gray-50"}>
                  {row.map((cell,j)=>(
                    <td key={j} className={`p-2.5 ${j===0?"font-semibold text-gray-700 text-left":"text-center text-gray-600"}`}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-400 mt-3">*Without polynomial/interaction features</p>
      </div>
    )
  },
  // WHEN TO USE WHAT
  {
    title: "When to Use What?",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">When to Use What? — Medical AI Guide</h2>
        <div className="grid grid-cols-2 gap-6">
          <div className="border-2 border-red-400 rounded-xl p-5">
            <div className="font-bold text-red-600 text-lg mb-2">Logistic Regression</div>
            <p className="text-sm text-gray-600 mb-2">Use when you need an <strong>interpretable baseline</strong>, especially for regulatory contexts or when clinicians must understand the model.</p>
            <div className="bg-red-50 rounded-lg p-2 text-xs text-gray-500">Ex: Risk scoring for clinical trials, identifying risk factors for disease</div>
          </div>
          <div className="border-2 border-green-400 rounded-xl p-5">
            <div className="font-bold text-green-600 text-lg mb-2">Random Forest</div>
            <p className="text-sm text-gray-600 mb-2">Use as a <strong>strong default</strong> for structured/tabular data. Great when you want robustness with minimal tuning.</p>
            <div className="bg-green-50 rounded-lg p-2 text-xs text-gray-500">Ex: EHR-based disease prediction, biomarker discovery, feature selection</div>
          </div>
          <div className="border-2 border-yellow-400 rounded-xl p-5">
            <div className="font-bold text-yellow-600 text-lg mb-2">Gradient Boosting</div>
            <p className="text-sm text-gray-600 mb-2">Use when you need <strong>maximum predictive accuracy</strong> on tabular data and can invest in hyperparameter tuning.</p>
            <div className="bg-yellow-50 rounded-lg p-2 text-xs text-gray-500">Ex: Kaggle competitions, precision medicine, outcome prediction from complex EHR data</div>
          </div>
          <div className="border-2 border-purple-400 rounded-xl p-5">
            <div className="font-bold text-purple-600 text-lg mb-2">MLP / Neural Network</div>
            <p className="text-sm text-gray-600 mb-2">Use when you have <strong>large datasets</strong>, complex patterns, or as a building block for deep learning architectures.</p>
            <div className="bg-purple-50 rounded-lg p-2 text-xs text-gray-500">Ex: Medical imaging (with CNNs), ECG analysis (with RNNs), multi-modal data fusion</div>
          </div>
        </div>
      </div>
    )
  },
  // SUMMARY
  {
    title: "Summary",
    content: (
      <div className="p-8">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Key Takeaways</h2>
        <div className="space-y-4">
          {[
            {c:"#e74c3c",t:"Logistic Regression",s:"Linear, interpretable, probabilistic. The go-to baseline for classification. Think of it as a single neuron with a sigmoid."},
            {c:"#27ae60",t:"Random Forests",s:"Ensemble of diverse trees via bagging + feature randomness. Robust, minimal tuning, built-in feature importance."},
            {c:"#f39c12",t:"Gradient Boosting",s:"Sequential ensemble that corrects errors iteratively. Often the best performer on tabular data, but requires careful tuning."},
            {c:"#8e44ad",t:"MLPs",s:"Stacked layers of neurons trained with backpropagation. Universal approximator. Foundation of deep learning. Best with large data."}
          ].map((item,i)=>(
            <div key={i} className="flex items-start gap-4 bg-white border rounded-xl p-4 shadow-sm">
              <div className="w-3 h-16 rounded-full flex-shrink-0" style={{background:item.c}}/>
              <div>
                <div className="font-bold text-lg" style={{color:item.c}}>{item.t}</div>
                <p className="text-gray-600 text-sm mt-1">{item.s}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-6 text-center text-gray-500 text-sm">
          Always start simple (logistic regression) → try ensembles (RF, XGBoost) → go deep (MLPs) if needed.
        </div>
      </div>
    )
  }
];

export default function SlideDeck() {
  const [idx, setIdx] = useState(0);
  const s = slides[idx];
  const hasBg = s.bg;

  const goPrev = useCallback(() => {
    setIdx((curr) => Math.max(0, curr - 1));
  }, []);

  const goNext = useCallback(() => {
    setIdx((curr) => Math.min(slides.length - 1, curr + 1));
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();

      const isEditable =
        target?.isContentEditable ||
        tag === "input" ||
        tag === "textarea" ||
        tag === "select";

      if (isEditable) return;

      if (e.key === "ArrowLeft" || e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        goPrev();
      } else if (e.key === "ArrowRight" || e.key === 'd' || e.key === 'D') {
        e.preventDefault();
        goNext();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goNext, goPrev]);

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Slide area */}
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-5xl bg-white rounded-2xl shadow-2xl overflow-hidden" style={{aspectRatio:"16/9", minHeight:480}}>
          <div className="h-full flex flex-col" style={{background: hasBg || "white"}}>
            <div className="flex-1 overflow-auto">
              {s.content}
            </div>
          </div>
        </div>
      </div>
      {/* Controls */}
      <div className="flex items-center justify-between px-8 py-3 bg-white border-t">
        <button onClick={goPrev} disabled={idx===0}
          className="px-5 py-2 bg-gray-800 text-white rounded-lg disabled:opacity-30 text-sm font-semibold hover:bg-gray-700 transition">
          ← Previous
        </button>
        <div className="text-sm text-gray-500">
          <span className="font-bold text-gray-800">{idx+1}</span> / {slides.length} — <span className="text-gray-400">{s.title}</span>
        </div>
        <button onClick={goNext} disabled={idx===slides.length-1}
          className="px-5 py-2 bg-gray-800 text-white rounded-lg disabled:opacity-30 text-sm font-semibold hover:bg-gray-700 transition">
          Next →
        </button>
      </div>
    </div>
  );
}