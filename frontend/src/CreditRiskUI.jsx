import React, { useState } from "react";
import { motion } from "framer-motion";
import "../src/styles/dashboard.css";

export default function CreditRiskUI() {

  const [formData, setFormData] = useState({
    loan_amnt: "",
    int_rate: "",
    annual_inc: "",
    dti: "",
    installment: "",
    revol_util: "",
    total_acc: "",
    grade: "",
    home_ownership: "",
    verification_status: "",
    purpose: "",
    term: ""
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // -------- Label Mapping --------
  const fieldLabels = {
    loan_amnt: "Loan Amount",
    int_rate: "Interest Rate (%)",
    annual_inc: "Annual Income",
    dti: "Debt-to-Income Ratio",
    installment: "Installment Amount",
    revol_util: "Revolving Credit Utilization",
    total_acc: "Total Credit Accounts",
    grade: "Credit Grade",
    home_ownership: "Home Ownership",
    verification_status: "Verification Status",
    purpose: "Loan Purpose",
    term: "Loan Term"
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // -------- Prediction --------
  const getPrediction = async () => {
    try {
      setLoading(true);

      const res = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData)
        }
      );

      const data = await res.json();
      setPrediction(data);

    } catch {
      alert("Prediction API failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="terminal-bg">

      {/* HEADER */}
      <div className="terminal-header">
        <h1>Credit Risk Intelligence Terminal</h1>
        <span>Production ML Monitoring System</span>
      </div>

      {/* GRID */}
      <div className="terminal-grid">

        {/* FORM PANEL */}
        <motion.div
          className="terminal-card"
          initial={{ opacity: 0, x: -40 }}
          animate={{ opacity: 1, x: 0 }}
        >

          <h2>Loan Application Inputs</h2>

          {/* HERO IMAGE */}
          <img
            src="https://images.unsplash.com/photo-1551288049-bebda4e38f71"
            className="hero-img"
          />

          {/* FORM */}
          <div className="form-grid">

            {Object.keys(formData).map((key) => (
              <div className="form-group" key={key}>

                <label className="form-label">
                  {fieldLabels[key]}
                </label>

                <input
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  className="form-input"
                  placeholder={`Enter ${fieldLabels[key]}`}
                />

              </div>
            ))}

          </div>

          <button
            className="btn-primary"
            onClick={getPrediction}
          >
            {loading ? "Analyzing Risk..." : "Run Credit Model"}
          </button>

        </motion.div>

        {/* RESULT PANEL */}
        <motion.div
          className="terminal-card"
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
        >

          <h2>Model Decision Engine</h2>

          {prediction ? (
            <>

              {/* KPI CARDS */}
              <div className="kpi-panel">

                <div className="kpi-box">
                  <span>Default Probability</span>
                  <h3>
                    {(prediction.default_probability * 100).toFixed(2)}%
                  </h3>
                </div>

                <div className="kpi-box">
                  <span>Risk Classification</span>
                  <h3 className={
                    prediction.default_prediction === 1
                      ? "risk-high"
                      : "risk-low"
                  }>
                    {prediction.default_prediction === 1
                      ? "HIGH RISK"
                      : "LOW RISK"}
                  </h3>
                </div>

              </div>

              {/* GAUGE */}
              <div className="gauge">
                <div
                  className="gauge-fill"
                  style={{
                    width:
                      `${prediction.default_probability * 100}%`
                  }}
                />
              </div>

            </>
          ) : (
            <p className="no-data">
              Awaiting prediction execution…
            </p>
          )}

          <button className="btn-danger">
            Trigger Monitoring & Retraining
          </button>

        </motion.div>

      </div>

      {/* FOOTER */}
      <div className="terminal-footer">
        Enterprise MLOps Lifecycle Console
      </div>

    </div>
  );
}
