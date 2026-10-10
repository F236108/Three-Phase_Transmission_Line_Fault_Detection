🚀 **Detecting Three-Phase Transmission Line Faults: From First-Principles Logistic Regression to Support Vector Machines** ⚡

How do high-voltage protective relays detect power system faults in milliseconds to prevent blackouts?

In our latest Applied Machine Learning project, we explored whether a linear decision boundary can detect transmission line faults from instantaneous 3-phase current ($I_a, I_b, I_c$) and voltage ($V_a, V_b, V_c$) measurements—and what happens when physics-driven non-linearities enter the picture.

🔍 **Key Highlights & Implementation**:
1️⃣ **First-Principles ML**: Built Binary Logistic Regression completely from scratch using pure NumPy (hypothesis, cost clipping to prevent overflow, and batch gradient descent).
2️⃣ **The Non-Separability Dilemma**: Visualizing the phase space ($I_a$ vs $I_b$) revealed that fault conditions form an annular ring around normal operation. As a result, linear models (Linear SVM & Raw Logistic Regression) plateau at ~74% accuracy with hundreds of dangerous missed faults.
3️⃣ **Two Elegant Solutions**:
   - **Domain Feature Engineering (Set Q)**: Introducing squared currents ($I^2$) and squared voltages ($V^2$) boosted Logistic Regression accuracy from **73.5% ➔ 98.8%**!
   - **RBF Kernel SVM**: Mapped features into high-dimensional space, hitting **99.7% test accuracy** with only 3 missed faults out of 2,401 test instances.
4️⃣ **Interactive Web UI**: Built a real-time Streamlit dashboard (`app/app.py`) for instantaneous inference.

💡 **Engineering Takeaway**: In protective relaying, a *missed fault* is catastrophic while a *false alarm* is an operational inconvenience. Physics-informed quadratic features allow simple linear models to achieve near-perfect performance with microsecond execution times suitable for embedded microcontrollers.

Check out the complete breakdown and code:
📝 **Read the full Medium Blog**: [Link to Medium Article]
💻 **GitHub Repository**: [Link to GitHub Repository]

#MachineLearning #PowerSystems #ElectricalEngineering #Python #DataScience #AppliedML #NumPy #SVM #DeepLearning #ArtificialIntelligence
