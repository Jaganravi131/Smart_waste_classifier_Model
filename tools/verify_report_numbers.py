#!/usr/bin/env python3
"""
verify_report_numbers.py - re-derives every arithmetic claim in the Phase I
report and prints PASS / FAIL for each one.

Run:
    python tools/verify_report_numbers.py

Nothing here needs a dataset or a model: it only checks that the numbers in the
document agree with each other.  Use it after editing the report.
"""

from __future__ import annotations

import math

BAR = "=" * 78
CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(title: str, computed, claimed, tolerance=0.01, unit="", note=""):
    """Compare a computed value against the value printed in the report."""
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if isinstance(computed, bool) or isinstance(claimed, bool):
        ok = computed == claimed
        fmt = lambda v: str(v)  # noqa: E731
    elif isinstance(computed, (int, float)) and isinstance(claimed, (int, float)):
        ok = abs(computed - claimed) <= tolerance
        fmt = lambda v: f"{v:,.4f}".rstrip("0").rstrip(".")  # noqa: E731
    else:
        ok = computed == claimed
        fmt = str
    CHECKS_PASSED += int(bool(ok))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {title}")
    print(f"        computed : {fmt(computed)}{unit}")
    print(f"        report   : {fmt(claimed)}{unit}")
    if note:
        print(f"        note     : {note}")
    print()
    return ok


def section(title: str):
    print(BAR)
    print(title)
    print(BAR)


def main() -> int:
    # ------------------------------------------------------------------
    section("1. Chapter 6 - test set size and headline accuracy")
    # Table 6.1 lists 6 rows (Paper is missing) x 80 images
    rows = {"Plastic": 79, "Metal": 78, "Glass": 76, "Medical waste": 76,
            "Organic": 75, "E-waste": 74}
    listed_images = 80 * len(rows)
    correct = sum(rows.values())
    print(f"Table 6.1 lists {len(rows)} classes x 80 images = {listed_images} images")
    print(f"Table 6.1 'classified' column sums to {correct}")
    print(f"{correct}/{listed_images} = {correct / listed_images * 100:.2f} %  "
          f"<-- this is exactly the 95.42 % quoted in the abstract")
    check("Table 6.1 overall accuracy", round(correct / listed_images * 100, 2), 95.42, 0.01, " %",
          "the quoted 95.42 % therefore excludes the missing Paper class")
    check("Test images listed in Table 6.1 vs the 7 classes of Table 5.4",
          listed_images, 560, note="Table 6.1 accounts for 480 of the 560 test images")
    print(f"95 % Wilson CI for {correct}/{listed_images}: ", end="")
    p = correct / listed_images
    z = 1.96
    n = listed_images
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    print(f"{100 * (centre - half):.2f} - {100 * (centre + half):.2f} %")
    print()

    section("2. Chapter 6 - the two accuracy tables disagree")
    # Table 6.1 per-class accuracy vs Table 6.2 recall for the same classes
    table61 = {"Plastic": 98.75, "Metal": 97.50, "Glass": 95.00, "Organic": 93.75}
    table62 = {"Plastic": 89.5, "Metal": 90.2, "Glass": 85.8, "Organic": 84.6}
    for cls in table61:
        check(f"{cls}: Table 6.1 accuracy vs Table 6.2 recall",
              table61[cls], table62[cls], 0.05, " %",
              "accuracy and recall are the same quantity for a single-label "
              "classifier evaluated class-by-class; they cannot differ by 8 points")
    macro_prec = (91.2 + 88.7 + 92.1 + 87.9 + 87.6) / 5
    macro_rec = (89.5 + 86.4 + 90.2 + 85.8 + 84.6) / 5
    check("Table 6.2 macro precision", round(macro_prec, 2), 89.5, 0.05, " %")
    check("Table 6.2 macro recall", round(macro_rec, 2), 87.3, 0.05, " %",
          "macro average covers only 5 of the 7 classes in Table 5.4")
    print(f"With macro recall {macro_rec:.1f} %, the overall accuracy cannot be 95.4 % unless "
          f"the test set is dominated by the strongest class - Table 6.1 says it is balanced.")
    print()

    section("3. Chapter 6 - detection log percentages")
    frames = 13_352
    detections = 12_810
    log = {"Paper": 12_601, "Glass": 149, "Cardboard": 26, "Plastic": 24, "Metal": 10}
    check("frame-level detection rate", round(detections / frames * 100, 2), 95.94, 0.01, " %")
    share_over_detections = {k: v / detections * 100 for k, v in log.items()}
    share_over_frames = {k: v / frames * 100 for k, v in log.items()}
    for cls, claimed_share in [("Paper", 94.38), ("Glass", 1.12), ("Cardboard", 0.19),
                               ("Plastic", 0.18), ("Metal", 0.07)]:
        check(f"Table 6.3 {cls} share computed over {detections} detections",
              round(share_over_detections[cls], 2), claimed_share, 0.005, " %",
              f"but over all {frames} frames it is {share_over_frames[cls]:.2f} %, "
              f"which is the value printed")
    print(f"Sum of the printed share column = "
          f"{sum([94.38, 1.12, 0.19, 0.18, 0.07]):.2f} % (should be 100 % of detections)")
    print()

    section("4. SDG chapter - the avoided-CO2 arithmetic")
    units, kg_per_day, organic_fraction = 45_000, 1.5, 0.5
    low, high = 0.4, 0.6  # t CO2e per tonne diverted
    organic_t_per_year = units * kg_per_day * 365 * organic_fraction / 1000
    avoided_low = organic_t_per_year * low
    avoided_high = organic_t_per_year * high
    print(f"organic waste diverted = {units:,} x {kg_per_day} kg/day x 365 x {organic_fraction} "
          f"= {organic_t_per_year:,.0f} t/yr")
    print(f"avoided = {avoided_low:,.0f} - {avoided_high:,.0f} t CO2e/yr")
    check("SS 2.2 avoided emissions", round((avoided_low + avoided_high) / 2), 3_300, 1, " t CO2e/yr",
          "the report's own inputs give 2.3x more than the stated 3,300 t")
    per_unit_kg = (avoided_low + avoided_high) / 2 * 1000 / units
    check("Appendix III per-unit avoided emissions", round(per_unit_kg), 73, 1, " kg CO2e/unit/yr")
    print()

    section("5. Chapter 4/5 - power budget vs the 31-hour endurance claim")
    wh = 12 * 6
    for hours in (24, 31):
        print(f"{wh} Wh over {hours} h allows {wh / hours:.2f} W average for the WHOLE unit")
    print("Raspberry Pi 4B alone draws 2.7 W idle / 3.4-6.4 W under load; the camera, the")
    print("380-lux LED ring, four ultrasonic sensors, the MQ-135 heater and 1,000 stepper")
    print("moves add to that. 2.3 W is below the Pi's idle draw.")
    check("R6 endurance implied by the 72 Wh pack", 24, 31, 0.5, " h",
          "needs a bigger pack or a Pi Zero / CM4-low-power design")
    print()

    section("6. Chapter 5/Appendix - stepper timing vs the 0.87 s cycle")
    steps_per_rev_out = 200 * 16 * 20       # 1/16 micro-step, 20:1 reduction
    steps_per_deg = steps_per_rev_out / 360
    steps_90 = 90 * steps_per_deg
    print(f"Steps per output revolution = 200 x 16 x 20 = {steps_per_rev_out:,}")
    print(f"Steps per degree = {steps_per_deg:.2f}; a 90 deg move = {steps_90:,.0f} steps")
    code_delay = 0.00015 * 2                # sleep before + after each pulse
    print(f"Appendix I.2 sleeps {code_delay * 1e6:.0f} us per step -> "
          f"{1 / code_delay:,.0f} steps/s -> {steps_90 * code_delay:.2f} s for 90 deg")
    check("90 deg move with the published code", round(steps_90 * code_delay, 2), 0.87, 0.05, " s",
          "4.8 s of stepping alone, before capture, inference and release")
    budget = 0.87 - 0.12 - 0.33
    print(f"To fit 0.87 s the motor must do 90 deg inside {budget:.2f} s -> "
          f"{steps_90 / budget:,.0f} steps/s -> {budget / steps_90 * 1e6:.1f} us/step")
    print("That is a 26 us step period from Python on a Pi 4 (realistically ~100-200 us")
    print("through RPi.GPIO), and 1/16 micro-stepping at that rate is not attainable.")
    print()

    section("7. Chapter 4 - sensors that cannot do what they are asked to do")
    print("HC-SR04 needs one trigger and one echo pin per sensor:"
          f" 4 sensors x 2 = 8 GPIO pins, Table 5.2 lists 4 (GPIO 5, 6, 13, 19).")
    check("GPIO pins needed for 4 ultrasonic fill sensors", 8, 4)
    print("Encoder resolution: 600 PPR quadrature = 2,400 counts/rev = 0.15 deg/count.")
    check("encoder count size", round(360 / 2400, 2), 0.15, 0.001, " deg")
    print("A 0.9 deg correction dead-band is 6 encoder counts wide, so the loop cannot")
    print("settle closer than ~0.9 deg; the quoted 1.7 deg worst case is therefore set by")
    print("the control law and the gear backlash, not by the 0.0056 deg micro-step claim.")
    print()

    section("8. Chapter 6.3 - architecture trade-off numbers")
    # EfficientNet-B3 baseline: 1.8 GFLOPs
    b3_flops = 1.8
    for name, flops, claimed_latency in [("B4", 4.2, 36), ("B5", 9.9, 62)]:
        check(f"EfficientNet-{name} extra computation vs B3", round((flops / b3_flops - 1) * 100),
              claimed_latency, 5, " %",
              "public FLOP counts: B0 0.39, B3 1.8, B4 4.2, B5 9.9 GFLOPs")
    check("EfficientNet-B3 forward pass, SS 5.6 (1.15 s) vs SS 6.3 (67 ms)",
          1150, 67, 30, " ms",
          "a 17x gap: the sweep and the deployed measurement cannot be the same device")
    print()

    section("9. Chapter 5.1 - bill of materials claims")
    bom = {"Raspberry Pi 4": 4900, "Camera v2": 1100, "NEMA 17": 1250, "DRV8825": 380,
           "Encoder": 1120, "Battery": 1850, "BMS": 620, "Buck": 410, "Moisture": 180,
           "Capacitive": 150, "Inductive": 520, "Ultrasonic x4": 480, "ADS1115": 320,
           "LED/buzzer": 330, "Discs": 1900, "Compartments": 1200, "Panels": 2400}
    total = sum(bom.values())
    check("BOM total", total, 19110, 0, " INR")
    check("100-unit COGS after the 32.5 % volume discount", round(total * 0.675), 12900, 1, " INR")
    compute = bom["Raspberry Pi 4"] + bom["Camera v2"]
    sensing = bom["Moisture"] + bom["Capacitive"] + bom["Inductive"] + bom["Ultrasonic x4"] + bom["ADS1115"]
    printed = bom["Discs"] + bom["Compartments"] + bom["Panels"]
    print(f"sensing + compute = {compute + sensing} = {(compute + sensing) / total * 100:.1f} % of the BOM")
    check("'sensing and compute are more than half the component cost'",
          round((compute + sensing) / total * 100, 1), 50, 5, " %")
    print(f"printed structure = {printed} = {printed / total * 100:.1f} % of the BOM")
    check("'the printed structure is less than a fifth'",
          round(printed / total * 100, 1), 20, 2, " %")
    print("Missing from the BOM although the text uses them: MQ-135 gas sensor, exhaust fan,")
    print("home/index sensor for the carousel, fuses, JST harnesses, fasteners.")
    print()

    section("10. Chapter 9 - market sizing")
    people, urban_share, per_household, households_per_device = 1.42e9, 0.35, 4.4, 12
    urban_households = people * urban_share / per_household
    devices = urban_households / households_per_device
    print(f"urban households = 1.42e9 x 0.35 / 4.4 = {urban_households:,.0f}")
    print(f"one device per 12 households  = {devices:,.0f} devices")
    check("SS 1.3.1 TAM 'users' for India", round(devices / 1e6, 1), 343, 5, " million",
          "the stated method yields ~9.4 million devices, not 343 million 'users'")
    som_households = 1.79e6 * 0.025
    check("SS 1.3.3 SOM household count", round(som_households), 44800, 100)
    check("SS 1.3.3 SOM value", round(som_households * 22500 / 1e7, 1), 100, 1, " crore")
    units_5y = 850 + 1050 + 1250 + 1500 + 1800
    check("units sold over 5 years (Table 9.6)", units_5y, 45000, 100,
          note="the plan sells 6,450 units, i.e. 14 % of the SOM it claims")
    revenue_5y_lakh = 201.50 + 245.70 + 295.00 + 354.00 + 424.80
    print(f"5-year revenue in Table 9.6 = {revenue_5y_lakh:,.2f} lakh = Rs {revenue_5y_lakh / 100:.2f} crore")
    print()

    section("11. Chapter 9 - profit and loss table")
    years = [1, 2, 3, 4, 5]
    units = [850, 1050, 1250, 1500, 1800]
    hardware = [191.25, 236.25, 281.25, 337.50, 405.00]
    subs = [10.25, 9.45, 13.75, 16.50, 19.80]
    cogs = [109.65, 132.30, 156.35, 185.10, 219.60]
    net = [16.90, 27.84, 38.35, 51.33, 71.37]
    for i, year in enumerate(years):
        check(f"Year {year} hardware revenue (units x Rs 22,500)",
              round(units[i] * 22500 / 1e5, 2), hardware[i], 0.02, " lakh")
    for i, year in enumerate(years):
        check(f"Year {year} net profit = gross - employee - marketing - ops - R&D",
              round(hardware[i] + subs[i] - cogs[i] - [38, 46, 54, 62, 72][i]
                    - [22, 24, 27, 31, 36][i] - [10.95, 11.56, 13.30, 16.57, 17.83][i]
                    - [4, 4, 6, 8, 8][i], 2), net[i], 0.05, " lakh")
    print("Subscription revenue is the only row that does not follow any stated driver:")
    cumulative = [sum(units[: i + 1]) for i in range(5)]
    for i, year in enumerate(years):
        implied = subs[i] * 1e5 / (3588 * cumulative[i])
        print(f"  Year {year}: {subs[i]:>5.2f} lakh with {cumulative[i]:>5,} units sold to date"
              f" -> implied attach rate {implied * 100:5.1f} %")
    check("subscription revenue grows with the installed base", subs[1] > subs[0], False,
          note="it falls from 10.25 to 9.45 lakh in Year 2 although units sold rise 24 %")
    print("Break-even: the report says 'breaks even during Year 2', but net profit is already")
    print(f"positive in Year 1 ({net[0]} lakh); cumulative Year 1+2 = {net[0] + net[1]:.2f} lakh"
          f" against a stated start-up cost of 8.00 lakh.")
    print()

    section("12. Report structure / reference list")
    issues = [
        "no in-text citation uses a number, so none of the 31 entries is actually linked",
        "references [11], [22], [27] are never cited anywhere in the text",
        "reference [15] has a placeholder article number: 'vol. 274, 123456'",
        "reference [18] cites the Solid Waste Management Rules 2026 - the notified rules are 2016",
        "reference [3] gives an article number (15987) in the issue field",
        "Appendix I lists 6 files that do not exist in the repository",
        "Fig. 6.1/6.2/6.3 and Fig. 6.5 were produced by scripts that generate random numbers",
    ]
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print()

    section("SUMMARY")
    print(f"{CHECKS_PASSED}/{CHECKS_TOTAL} arithmetic checks pass")
    print("Everything marked FAIL is an error that a viva examiner can reproduce with a")
    print("calculator, which is why these matter more than the typographical ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
