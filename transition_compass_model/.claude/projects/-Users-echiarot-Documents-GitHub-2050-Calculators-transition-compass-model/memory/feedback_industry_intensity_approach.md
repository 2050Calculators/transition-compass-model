---
name: feedback-industry-intensity-approach
description: User prefers deriving Swiss energy intensities from Swiss sources directly, not calibrating/scaling EU27 values
metadata:
  type: feedback
---

Do not derive correction coefficients by calibrating against SFOE energy data (e.g. computing kt = SFOE_energy / intensity to back-fill production). Coefficients must be physically grounded — price ratios, employment statistics, production statistics — independent of SFOE.

**Why:** SFOE calibration makes model outputs circular (the correction guarantees the output matches the target, not that the input is physically correct). The user wants inputs (production kt, energy intensities) to be derivable from physical data, with SFOE used only as a validation check.

**How to apply:**
- For production kt corrections (e.g. k_mae): ground in price ratios (actual CHF/t vs IO proxy CHF/t), employment-based estimates, or external production statistics. Never compute as energy_gap / intensity.
- For energy intensities: derive from Swiss data (SFOE energy / Swiss production) when possible.
- Code comments must state the physical reasoning (price ratio, data source), not reference SFOE TWh values as the calibration target.
- Current example: k_mae = 24 because IO uses electronics proxy (~50,000 CHF/t) but MAE is dominated by fabricated metals (~2,000 CHF/t); ratio = 25, conservative estimate = 24.
