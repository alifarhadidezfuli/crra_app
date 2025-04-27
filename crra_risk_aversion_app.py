import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

st.set_page_config(page_title="Risk Aversion Calculator", page_icon="🎲", layout="centered")

st.title("🎲 Risk Aversion Calculator – Behavioral Economics")
st.write("This app calculates the relative risk aversion coefficient (Gamma) based on a CRRA utility function.")

st.header("Input Parameters")
Wce = st.number_input("Certainty Equivalent Value (Wce)", min_value=0.0, value=20000.0)
p1 = st.number_input("Probability of Winning (between 0 and 1)", min_value=0.0, max_value=1.0, value=0.5)
gain = st.number_input("Gain Amount if Win", value=110.0)
loss = st.number_input("Loss Amount if Lose", value=100.0)

calculate_button = st.button("Calculate Gamma 🎯")

if calculate_button:
    p2 = 1 - p1
    outcome1 = Wce + gain
    outcome2 = Wce - loss

    def utility(x, gamma):
        if gamma == 1:
            return np.log(x)
        else:
            return (x**(1-gamma)) / (1-gamma)

    def objective(gamma):
        expected_utility = p1 * utility(outcome1, gamma) + p2 * utility(outcome2, gamma)
        ce_utility = utility(Wce, gamma)
        return expected_utility - ce_utility

    max_limit = 1000
    upper_bound = 10

    found = False
    while upper_bound <= max_limit:
        f_a = objective(0)
        f_b = objective(upper_bound)

        if f_a * f_b < 0:
            gamma_solution = opt.root_scalar(objective, bracket=[0, upper_bound], method='brentq')
            if gamma_solution.converged:
                gamma = gamma_solution.root

                U_win = utility(outcome1, gamma)
                U_loss = utility(outcome2, gamma)
                U_lott = p1 * U_win + p2 * U_loss
                U_Wce = utility(Wce, gamma)

                st.success(f"Relative Risk Aversion Coefficient (Gamma) = {gamma:.4f}")
                st.info(f"U(win) = {U_win:.6e}")
                st.info(f"U(loss) = {U_loss:.6e}")
                st.info(f"U(lott) = {U_lott:.6e}")
                st.info(f"U(Wce) = {U_Wce:.6e}")

                st.header("📋 Risk Behavior Interpretation")
                if gamma < 1:
                    st.write("🔵 The individual is approximately risk-neutral.")
                elif 1 <= gamma <= 3:
                    st.write("🟡 The individual is moderately risk-averse.")
                else:
                    st.write("🔴 The individual is highly risk-averse.")

                # نمودار اختلاف مطلوبیت نسبت به گاما
                gammas = np.linspace(0.01, gamma * 2, 300)
                differences = [objective(g) for g in gammas]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(gammas, differences, label='U(lott) - U(Wce)', color='royalblue', linewidth=2)
                ax.axhline(0, color='red', linestyle='--', label='Zero Line')
                ax.scatter([gamma], [0], color='limegreen', edgecolors='black', s=100, label=f'Gamma Found ({gamma:.2f})')
                ax.set_title('Difference in Utility vs Gamma', fontsize=16, fontweight='bold')
                ax.set_xlabel('Gamma', fontsize=14)
                ax.set_ylabel('Difference in Utility', fontsize=14)
                ax.legend()
                ax.grid(True, linestyle=':', linewidth=0.7)
                st.pyplot(fig)

                found = True
                break
        else:
            upper_bound *= 2

    if not found:
        st.error("❗ Could not find Gamma. Please check your inputs.")