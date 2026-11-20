import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

H_FILL = PatternFill("solid", fgColor="2E4057")
S_FILL = PatternFill("solid", fgColor="4A6FA5")
P_FILL = PatternFill("solid", fgColor="F0F4F8")
B_FILL = PatternFill("solid", fgColor="D6E4F0")
H_FONT = Font(color="FFFFFF", bold=True, name="Calibri", size=11)
S_FONT = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
P_FONT = Font(bold=True, name="Calibri", size=11)
B_FONT = Font(bold=True, name="Calibri", size=11)
N_FONT = Font(name="Calibri", size=11)
I_FONT = Font(italic=True, color="808080", name="Calibri", size=10)
LB_FONT = Font(bold=True, name="Calibri", size=10)
TX_FONT = Font(name="Calibri", size=10)
IT_FONT = Font(italic=True, color="606060", name="Calibri", size=10)
WRAP = Alignment(wrap_text=True, vertical="top")

MAIN_H = [
    "Product",
    "Component",
    "Material / Mineral",
    "Intermediate Material / Mineral",
    "Raw mineral",
    "Modelled techs",
]
MAIN_W = [22, 54, 26, 32, 50, 65]
MAP_H = ["Name", "Model name", "Production techs", "Notes"]
MAP_W = [25, 20, 65, 52]


def write_main(wb, title, rows):
    ws = wb.create_sheet(title=title)
    for c, h in enumerate(MAIN_H, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = H_FILL
        cell.font = H_FONT
        cell.alignment = WRAP
    r = 2
    for row in rows:
        prod, comp, mat, interm, raw, tech = (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )
        bm = row[6] if len(row) > 6 else False
        bi = row[7] if len(row) > 7 else False
        note = row[8] if len(row) > 8 else False
        pfirst = row[9] if len(row) > 9 else False
        for c, val in enumerate([prod, comp, mat, interm, raw, tech], 1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.alignment = WRAP
            if note:
                cell.font = I_FONT
            elif c == 3 and bm:
                cell.fill = B_FILL
                cell.font = B_FONT
            elif c == 4 and bi:
                cell.fill = B_FILL
                cell.font = B_FONT
            elif c == 1 and pfirst and prod:
                cell.fill = P_FILL
                cell.font = P_FONT
            else:
                cell.font = N_FONT
        r += 1
    for c, w in enumerate(MAIN_W, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 15
    return ws


def write_map(wb, title, rows):
    ws = wb.create_sheet(title=title)
    for c, h in enumerate(MAP_H, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = H_FILL
        cell.font = H_FONT
        cell.alignment = WRAP
    r = 2
    for row in rows:
        note = row[4] if len(row) > 4 else False
        bold = row[5] if len(row) > 5 else False
        for c, val in enumerate(row[:4], 1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.alignment = WRAP
            if note:
                cell.font = I_FONT
            elif bold and c <= 3:
                cell.fill = B_FILL
                cell.font = B_FONT
            else:
                cell.font = N_FONT
        r += 1
    for c, w in enumerate(MAP_W, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    return ws


def write_readme(wb, title):
    ws = wb.create_sheet(title=title)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 90
    rows = [
        ("hdr", "MATERIAL CHAIN — REFERENCE GUIDE", ""),
        ("blank", "", ""),
        ("sub", "PURPOSE", ""),
        (
            "body",
            "",
            "Maps product variables to their full material chain: component → material → intermediate → raw mineral. Used as reference for PRODCOM mapping and material flow tracking.",
        ),
        ("blank", "", ""),
        ("sub", "COLUMN STRUCTURE", ""),
        (
            "lbl",
            "Product",
            "Product variable name (e.g. LDV_BEV, buildings, fertilizer)",
        ),
        (
            "lbl",
            "Component",
            "What the material becomes in the finished product — '[Material] products (forms)' style",
        ),
        (
            "lbl",
            "Material / Mineral",
            "Model name of the material or mineral tracked (see Material name mapping sheet for full list)",
        ),
        (
            "lbl",
            "Intermediate Material / Mineral",
            "Physical input that combines to form the Material / Mineral — either a material (Cement → Concrete; Ammonia → Nitrogen fertilizer; Pulp → Paper/paperboard) or a mineral (Non-metallic minerals → Concrete). Bold if it has a model name; Pulp is an exception — shown for context but NOT bold.",
        ),
        ("lbl", "Raw mineral", "Extracted minerals / feedstocks"),
        (
            "lbl",
            "Modelled techs",
            "Production technology names; '—' means tracked for mass/EOL balance only, no production tech",
        ),
        ("blank", "", ""),
        ("sub", "BOLD CONVENTION", ""),
        (
            "body",
            "",
            "Bold + blue highlight = material has a model name and is tracked in the model.",
        ),
        (
            "body",
            "",
            "Production tech may be '—' for mass-tracking-only categories (nmm, met, other) — these are still bold.",
        ),
        (
            "note",
            "",
            "NOT bold: Concrete, Nitrogen fertilizer (demand-side composites — not direct production outputs).",
        ),
        (
            "note",
            "",
            "NOT bold: Pulp in Intermediate column (paper-tech covers the full C17 chain including pulping).",
        ),
        ("blank", "", ""),
        ("sub", "MATERIAL CATEGORIES", ""),
        (
            "lbl",
            "Modelled (have tech)",
            "steel · aluminium · glass · copper · chem · paper · timber · asphalt · cement · ammonia · lime · ois",
        ),
        ("lbl", "Mass/EOL only (no tech)", "nmm · met · other"),
        ("blank", "", ""),
        (
            "lbl",
            "nmm",
            "Non-metallic minerals: aggregate (sand, gravel, crushed stone), natural graphite. Eurostat EW-MFA 'Non-metallic minerals'. NACE B08.",
        ),
        (
            "lbl",
            "met",
            "Metallic minerals: lead, zinc, lithium, cobalt, PGMs, rare earths, nickel in batteries. Eurostat EW-MFA 'Metal ores'. NACE B07.",
        ),
        (
            "lbl",
            "other",
            "Other materials: HFC refrigerants, synthetic rubber, undisaggregated manufactured assemblies. Residual category.",
        ),
        (
            "lbl",
            "ois",
            "OIS (Other Industry Sectors): manufactured ceramics, bricks, tiles, gypsum board (NACE C23). HAS a production tech: ois-tech / ois-sec.",
        ),
        ("blank", "", ""),
        ("sub", "KEY RULES", ""),
        (
            "lbl",
            "1. Aggregate",
            "Goes in Intermediate material column (not Material) — physical input to Concrete alongside Cement.",
        ),
        (
            "lbl",
            "2. Ceramics → OIS",
            "Ceramics go to ois-tech, not nmm. They are manufactured NACE C23 products, not mined minerals.",
        ),
        (
            "lbl",
            "3. Metal ores (minor)",
            "Lead, zinc, Li, Co, PGMs, rare earths → met. Not tracked through any named production tech.",
        ),
        (
            "lbl",
            "4. Construction minerals",
            "Sand, gravel, crushed stone, natural graphite → nmm. NACE B08 quarrying.",
        ),
        (
            "lbl",
            "5. HFC / rubber / misc",
            "HFC refrigerants, rubber seals, undisaggregated assemblies → other. Not a mineral or metal ore.",
        ),
        ("blank", "", ""),
        ("sub", "HOW TO REBUILD", ""),
        (
            "body",
            "",
            "Run rebuild_material_chain.py located in the same directory as this file. Requires openpyxl.",
        ),
    ]
    r = 1
    for style, a, b in rows:
        if style == "hdr":
            for c, val in enumerate([a, b], 1):
                cell = ws.cell(row=r, column=c, value=val)
                cell.fill = H_FILL
                cell.font = H_FONT
                cell.alignment = WRAP
        elif style == "sub":
            for c, val in enumerate([a, b], 1):
                cell = ws.cell(row=r, column=c, value=val)
                cell.fill = S_FILL
                cell.font = S_FONT
                cell.alignment = WRAP
        elif style == "lbl":
            cell_a = ws.cell(row=r, column=1, value=a)
            cell_a.font = LB_FONT
            cell_a.alignment = WRAP
            cell_b = ws.cell(row=r, column=2, value=b)
            cell_b.font = TX_FONT
            cell_b.alignment = WRAP
        elif style == "body":
            ws.cell(row=r, column=1, value=a).font = TX_FONT
            cell_b = ws.cell(row=r, column=2, value=b)
            cell_b.font = TX_FONT
            cell_b.alignment = WRAP
        elif style == "note":
            cell_b = ws.cell(row=r, column=2, value=b)
            cell_b.font = IT_FONT
            cell_b.alignment = WRAP
        r += 1
    return ws


def note(text):
    return (None, text, None, None, None, None, False, False, True, False)


# ── MAPPING ──────────────────────────────────────────────────────────────────
mapping = [
    (
        "Crude steel",
        "steel",
        "steel-BF-BOF / steel-hisarna / steel-hydrog-DRI / steel-scrap-EAF",
        "",
        False,
        True,
    ),
    ("Aluminium ingot", "aluminium", "aluminium-prim / aluminium-sec", "", False, True),
    ("Flat glass", "glass", "glass-glass / glass-sec", "", False, True),
    ("Refined copper", "copper", "copper-tech / copper-sec", "", False, True),
    ("Polymers / resins", "chem", "chem-chem-tech / chem-sec", "", False, True),
    ("Paper / paperboard", "paper", "paper-tech / paper-sec", "", False, True),
    ("Sawnwood", "timber", "timber-tech / timber-sec", "", False, True),
    ("Asphalt", "asphalt", "asphalt-tech / asphalt-sec", "", False, True),
    (
        "Cement",
        "cement",
        "cement-dry-kiln / cement-wet-kiln / cement-geopolym / cement-sec-post-consumer",
        "Intermediate for Concrete (combines with aggregate)",
        False,
        True,
    ),
    (
        "Ammonia",
        "ammonia",
        "ammonia-tech",
        "Intermediate for Nitrogen fertilizer (Haber-Bosch → urea, AN, AS)",
        False,
        True,
    ),
    ("Lime", "lime", "lime-lime", "", False, True),
    (
        "OIS",
        "ois",
        "ois-tech / ois-sec",
        "Other Industry Sectors — manufactured ceramics, bricks, tiles, gypsum board (NACE C23)",
        False,
        True,
    ),
    (
        "Non-metallic minerals",
        "nmm",
        "—",
        "Construction + industrial minerals: aggregate (sand, gravel, crushed stone), natural graphite. Mass/EOL tracking only — no production tech. Aligns with Eurostat EW-MFA 'Non-metallic minerals'.",
        False,
        True,
    ),
    (
        "Metallic minerals",
        "met",
        "—",
        "Metal ores not tracked through a named production tech: lead, zinc, lithium, cobalt, PGMs, rare earths, nickel (in batteries). Mass/EOL tracking only. Aligns with Eurostat EW-MFA 'Metal ores'.",
        False,
        True,
    ),
    (
        "Other materials",
        "other",
        "—",
        "Processed materials not mapping to a mineral or metal category: HFC refrigerants, synthetic rubber, undisaggregated manufactured assemblies. Residual — mass/EOL tracking only.",
        False,
        True,
    ),
    (None, None, None, None, False, False),
    (
        "Concrete-and-inert",
        "—",
        "—",
        "Demand-driven composite (Cement + aggregate); production tracked via Cement",
        False,
        False,
    ),
    (None, None, None, None, False, False),
    (
        "Bold cell = material / mineral with a model name (tracked in the model; production tech may be '—' for mass-tracking-only categories)",
        "",
        "",
        "",
        True,
        False,
    ),
    (
        "Intermediate col bold = Cement (→ Concrete) and Ammonia (→ N-fertilizer)",
        "",
        "",
        "",
        True,
        False,
    ),
    (
        "Pulp shown as Intermediate for paper rows but NOT bold — paper-tech covers full C17 chain",
        "",
        "",
        "",
        True,
        False,
    ),
]

# ── BUILDINGS ─────────────────────────────────────────────────────────────────
buildings = [
    (
        "Buildings",
        "Concrete (structural frame, walls, floors)",
        "Concrete",
        "Cement",
        "Limestone + Clay + Gypsum",
        "cement-dry-kiln / cement-wet-kiln / cement-geopolym / cement-sec-post-consumer",
        False,
        True,
        False,
        True,
    ),
    (
        None,
        None,
        None,
        "Non-metallic minerals",
        "Sand + Gravel + Crushed stone",
        "(not modelled — NACE B08 mining)",
        False,
        True,
        False,
        False,
    ),
    (
        None,
        "Flat glass (windows, curtain walls, insulating units)",
        "Flat glass",
        None,
        "Silica sand + Soda ash + Limestone + Dolomite",
        "glass-glass / glass-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Steel products (rebar, sections, sheets)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-hisarna / steel-hydrog-DRI / steel-scrap-EAF",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Aluminium products (profiles, frames, cladding)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Lime (mortar, plaster, whitewash)",
        "Lime",
        None,
        "Limestone",
        "lime-lime",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Sawnwood (structural frame, joists, cladding)",
        "Sawnwood",
        None,
        "Roundwood (A02 forestry)",
        "timber-tech / timber-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (wire, plumbing pipes)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Asphalt (roofing membrane, waterproofing)",
        "Asphalt",
        None,
        "Crude oil",
        "asphalt-tech / asphalt-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Plastic / chemical products (insulation, adhesives, PVC)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Paper products (plasterboard facing, vapour barriers)",
        "Paper / paperboard",
        "Pulp",
        "Roundwood (A02 forestry)",
        "paper-tech / paper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "OIS products (bricks, tiles, gypsum board)",
        "OIS",
        None,
        "Clay (dominant) + Gypsum + Natural stone",
        "ois-tech / ois-sec",
        True,
        False,
        False,
        False,
    ),
]


# ── VEHICLES ──────────────────────────────────────────────────────────────────
def vrow(prod, comp, mat, interm, raw, tech, bm, bi, is_note=False, pfirst=False):
    return (prod, comp, mat, interm, raw, tech, bm, bi, is_note, pfirst)


ST = (
    "Crude steel",
    "Iron ore + Coking coal",
    "steel-BF-BOF / steel-hisarna / steel-hydrog-DRI / steel-scrap-EAF",
)
AL = ("Aluminium ingot", "Bauxite", "aluminium-prim / aluminium-sec")
PL = ("Polymers / resins", "Crude oil + Natural gas", "chem-chem-tech / chem-sec")
GL = ("Flat glass", "Silica sand + Soda ash + Limestone", "glass-glass / glass-sec")
CU = ("Refined copper", "Copper ore", "copper-tech / copper-sec")
TI = ("Sawnwood", "Roundwood (A02 forestry)", "timber-tech / timber-sec")


def s(comp):
    return vrow(
        None, f"Steel products ({comp})", ST[0], None, ST[1], ST[2], True, False
    )


def a(comp):
    return vrow(
        None, f"Aluminium products ({comp})", AL[0], None, AL[1], AL[2], True, False
    )


def p(comp):
    return vrow(
        None,
        f"Plastic / chemical products ({comp})",
        PL[0],
        None,
        PL[1],
        PL[2],
        True,
        False,
    )


def g():
    return vrow(
        None, "Flat glass (windscreen, windows)", GL[0], None, GL[1], GL[2], True, False
    )


def cu(comp):
    return vrow(
        None, f"Copper products ({comp})", CU[0], None, CU[1], CU[2], True, False
    )


def ti(comp):
    return vrow(None, f"Sawnwood ({comp})", TI[0], None, TI[1], TI[2], True, False)


def nmm(comp, raw):
    return vrow(
        None,
        f"Non-metallic minerals ({comp})",
        "Non-metallic minerals",
        None,
        raw,
        "(not modelled — NACE B mining)",
        True,
        False,
    )


def met(comp, raw):
    return vrow(
        None,
        f"Metallic minerals ({comp})",
        "Metallic minerals",
        None,
        raw,
        "(not modelled — NACE B mining)",
        True,
        False,
    )


def otm(comp, raw):
    return vrow(
        None,
        f"Other materials ({comp})",
        "Other materials",
        None,
        raw,
        "(not modelled)",
        True,
        False,
    )


def ois_row(comp, raw="Clay + Alumina + mixed"):
    return vrow(
        None,
        f"OIS products ({comp})",
        "OIS",
        None,
        raw,
        "ois-tech / ois-sec",
        True,
        False,
    )


vehicles = [
    # LDV_ICE-gasoline
    vrow(
        "LDV_ICE-gasoline",
        "Steel products (body, chassis, engine block)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("engine block, closures, wheels"),
    p("bumpers, trims, hoses, tyres"),
    g(),
    cu("wiring, connectors"),
    met("lead 12V battery, zinc, brass fittings", "Lead ore + Zinc ore + mixed"),
    note("LDV_ICE-gas is a processor alias with identical composition"),
    # LDV_ICE-diesel
    vrow(
        "LDV_ICE-diesel",
        "Steel products (body, chassis, diesel engine block — slightly heavier)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("engine block, closures, wheels"),
    p("bumpers, trims, hoses, tyres"),
    g(),
    cu("wiring, connectors"),
    met("lead 12V battery, zinc, brass fittings", "Lead ore + Zinc ore + mixed"),
    # LDV_PHEV-gasoline
    vrow(
        "LDV_PHEV-gasoline",
        "Steel products (body, chassis, e-motor frame)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("body panels, motor housing"),
    p("bumpers, trims, hoses"),
    g(),
    cu("motor windings, wiring — more than ICE"),
    met("NiMH battery (nickel, cobalt, iron)", "Nickel ore + Cobalt ore + Iron ore"),
    met("lead, zinc", "Lead ore + Zinc ore"),
    note(
        "Uses LDV_HEV literature data as proxy; NiMH battery mass NOT tracked separately"
    ),
    # LDV_BEV
    vrow(
        "LDV_BEV",
        "Steel products (body, chassis, e-motor stator)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("structural parts, motor housing"),
    p("body components, rubber"),
    g(),
    cu("motor windings, wiring — more than ICE"),
    met("lead, zinc, brass fittings", "Lead ore + Zinc ore + mixed"),
    note(
        "Battery mass tracked separately — see battery-lion-LDV_BEV on Batteries sheet"
    ),
    # LDV_FCEV
    vrow(
        "LDV_FCEV",
        "Steel products (body, chassis, fuel cell frame)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("fuel cell bipolar plates, body panels"),
    p("CFRP H2 pressure vessel, polymer membrane, interior"),
    g(),
    cu("fuel cell wiring, motor"),
    met("platinum catalyst, nickel, carbon GDL", "PGM ore + Nickel ore + Graphite"),
    otm("rubber seals, polymer membrane", "Synthetic rubber + mixed"),
    # HDV_ICE-diesel
    vrow(
        "HDV_ICE-diesel",
        "Steel products (chassis, cab, engine, axles)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("cab body, engine block, radiator, wheels"),
    ti("trailer floor, interior panelling"),
    p("rubber: tyres, hoses; GFRP body panels; plastics: interior"),
    g(),
    ois_row(
        "ceramic brake pads, catalytic converter substrates",
        "Clay + Alumina + Cordierite",
    ),
    otm("brass fittings, paint, leather, stainless trim", "Mixed"),
    note(
        "Avg of HDVH Class 8 (day-cab + sleeper) + HDVM Class 6 PnD; HDV_ICE-gas and HDV_ICE-gasoline are processor aliases"
    ),
    # HDV_PHEV-diesel
    vrow(
        "HDV_PHEV-diesel",
        "Steel products (chassis, cab, e-motor frame)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("cab body, motor, radiator"),
    ti("trailer floor"),
    p("rubber + GFRP + interior plastics"),
    g(),
    met("NiMH battery (nickel, cobalt, iron)", "Nickel ore + Cobalt ore + Iron ore"),
    ois_row("ceramic brake components", "Clay + Alumina"),
    otm("brass fittings, paint, stainless trim", "Mixed"),
    note(
        "Avg of HDVH/M HEV variants; NiMH battery NOT tracked separately; HDV_PHEV-gasoline is a processor alias"
    ),
    # HDV_BEV
    vrow(
        "HDV_BEV",
        "Steel products (chassis, cab, e-motor frame)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("cab, motor housing, radiator"),
    ti("trailer floor"),
    p("rubber + GFRP + interior plastics"),
    g(),
    ois_row("ceramic brake components", "Clay + Alumina"),
    otm("brass fittings, paint", "Mixed"),
    note(
        "Battery mass tracked separately — see battery-lion-HDV_BEV on Batteries sheet; Avg of HDVH/M BEV variants"
    ),
    # HDV_FCEV
    vrow(
        "HDV_FCEV",
        "Steel products (chassis, cab, fuel cell frame, stainless bipolar plates)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("cab body, motor, bipolar plates"),
    p("CFRP H2 vessel + GFRP body panels + polymer membrane"),
    ti("trailer floor"),
    g(),
    ois_row("ceramic cell components", "Clay + Alumina"),
    met("platinum catalyst, carbon GDL", "PGM ore + Graphite + mixed"),
    # ships_ICE
    vrow(
        "ships_ICE",
        "Steel products (hull, bulkheads, structural frames — ~12% of unit mass)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("superstructure"),
    p("fittings, insulation composites"),
    cu("wiring, electrical systems"),
    otm(
        "ship machinery, propulsion, outfitting (~85% of hull mass — not disaggregated in literature)",
        "Mixed",
    ),
    note("Variable ships_ICE should be renamed ships_ICE-diesel — naming fix pending"),
    # trains_CEV
    vrow(
        "trains_CEV",
        "Steel products (car body, bogies, underframe)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("car body panels, interior"),
    p("interior, insulation composites"),
    cu("traction motors, electrical"),
    otm("braking, HVAC, pantograph, seats, glass", "Mixed"),
    note(
        "Processor overrides raw literature (full freight consist) with weighted avg: passenger 310 t (w=0.557) + freight 2,788 t (w=0.443) EU-configs"
    ),
    # planes_ICE
    vrow(
        "planes_ICE",
        "Steel products (landing gear, engine pylons, fasteners)",
        ST[0],
        None,
        ST[1],
        ST[2],
        True,
        False,
        False,
        True,
    ),
    a("fuselage skin, wings, structural frames"),
    p("CFRP fuselage composites, interior panels"),
    cu("avionics, wiring harness"),
    met("titanium alloys, superalloys, nickel", "Titanium ore + Nickel ore + mixed"),
    otm("tyres, anti-corrosion coatings", "Synthetic rubber + mixed"),
    note(
        "aviation_BEV and aviation_H2 use planes_ICE as material proxy (same airframe, different propulsion)"
    ),
    # aliases / pending
    (
        "bus_ICE-diesel / bus_ICE-gas / bus_ICE-gasoline",
        "Same material chain as HDV_ICE-diesel (processor copies of model row)",
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        True,
    ),
    (
        "bus_PHEV-diesel / bus_PHEV-gasoline",
        "Same material chain as HDV_PHEV-diesel (processor copies of model row)",
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        True,
    ),
    (
        "bus_BEV",
        "Same material chain as HDV_BEV (processor copy of model row)",
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        True,
    ),
    (
        "bus_FCEV",
        "Same material chain as HDV_FCEV (processor copy of model row)",
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        True,
    ),
    (
        "trains_ICE-diesel",
        "Same material chain as trains_CEV (processor copy of weighted-avg row — no structural difference modelled)",
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        True,
    ),
    (
        "2W",
        "TODO: not modelled — no material decomposition in preprocessing or literature review",
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        True,
    ),
    (
        "bus_CEV",
        "In progress — not yet in preprocessing",
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        True,
    ),
]


# ── BATTERIES ────────────────────────────────────────────────────────────────
def bat_rows(prod):
    return [
        (
            prod,
            "Aluminium products (cell casing, cathode current collector)",
            "Aluminium ingot",
            None,
            "Bauxite",
            "aluminium-prim / aluminium-sec",
            True,
            False,
            False,
            True,
        ),
        (
            None,
            "Steel products (cell housing)",
            "Crude steel",
            None,
            "Iron ore + Coking coal",
            "steel-BF-BOF / steel-scrap-EAF",
            True,
            False,
            False,
            False,
        ),
        (
            None,
            "Plastic products (separator, electrolyte packaging)",
            "Polymers / resins",
            None,
            "Crude oil + Natural gas",
            "chem-chem-tech / chem-sec",
            True,
            False,
            False,
            False,
        ),
        (
            None,
            "Copper products (anode current collector)",
            "Refined copper",
            None,
            "Copper ore",
            "copper-tech / copper-sec",
            True,
            False,
            False,
            False,
        ),
        (
            None,
            "Metallic minerals (NMC cathode: lithium, cobalt, nickel, manganese)",
            "Metallic minerals",
            None,
            "Lithium brine/spodumene + Cobalt ore + Nickel ore + Manganese ore",
            "(not modelled — NACE B mining)",
            True,
            False,
            False,
            False,
        ),
        (
            None,
            "Non-metallic minerals (graphite anode)",
            "Non-metallic minerals",
            None,
            "Natural graphite (NACE B) / synthetic graphite (NACE C20)",
            "(not modelled)",
            True,
            False,
            False,
            False,
        ),
    ]


batteries = (
    bat_rows("battery-lion-LDV_BEV")
    + bat_rows("battery-lion-LDV_PHEV")
    + bat_rows("battery-lion-HDV_BEV")
    + bat_rows("battery-lion-HDV_PHEV")
    + [
        note(
            "NiMH and Pb-Ac batteries have literature data but are NOT separate variables — their mass is included in vehicle glider entries (LDV_HEV and HDV_HEV)"
        )
    ]
)

# ── PACKAGING & PAPER ────────────────────────────────────────────────────────
packaging = [
    (
        "aluminium-pack",
        "Aluminium products (can, foil, container)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        True,
    ),
    (
        "glass-pack",
        "Flat glass (bottle, jar, container)",
        "Flat glass",
        None,
        "Silica sand + Soda ash + Limestone + Cullet",
        "glass-glass / glass-sec",
        True,
        False,
        False,
        True,
    ),
    (
        "paper-pack",
        "Paper products (corrugated box, cartonboard)",
        "Paper / paperboard",
        "Pulp",
        "Roundwood (A02 forestry)",
        "paper-tech / paper-sec",
        True,
        False,
        False,
        True,
    ),
    (
        "paper-print",
        "Paper products (printing, graphic paper)",
        "Paper / paperboard",
        "Pulp",
        "Roundwood (A02 forestry)",
        "paper-tech / paper-sec",
        True,
        False,
        False,
        True,
    ),
    (
        "paper-san",
        "Paper products (tissue, sanitary, household paper)",
        "Paper / paperboard",
        "Pulp",
        "Roundwood (A02 forestry)",
        "paper-tech / paper-sec",
        True,
        False,
        False,
        True,
    ),
    (
        "plastic-pack",
        "Plastic / chemical products (film, bottle, container: PET, HDPE, PP)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        True,
    ),
]

# ── APPLIANCES ───────────────────────────────────────────────────────────────
appliances = [
    # computer
    (
        "computer",
        "Plastic / chemical products (casing, keyboard, circuit boards: ABS/PMMA)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Aluminium products (chassis, heatsink)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Flat glass (display panel, screen)",
        "Flat glass",
        None,
        "Silica sand + ITO coating substrate",
        "glass-glass / glass-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Steel products (structural parts)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (PCB wiring, connectors)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Metallic minerals (Li battery, Mg frame, rare earth minerals, semiconductor trace metals)",
        "Metallic minerals",
        None,
        "Lithium brine + Mg ore + Silicon + Rare earth minerals",
        "(not modelled — NACE B mining)",
        True,
        False,
        False,
        False,
    ),
    note(
        "Semiconductor wafer production not modelled; Li battery mass included in the product variable — not tracked separately"
    ),
    # dryer
    (
        "dryer",
        "Steel products (drum, casing, motor frame)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Plastic / chemical products (control panel, door, liner)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Aluminium products (motor parts, heat exchanger)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (motor windings)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Sawnwood (internal frame / packing)",
        "Sawnwood",
        None,
        "Roundwood (A02 forestry)",
        "timber-tech / timber-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Other materials (rubber: door seal, hoses)",
        "Other materials",
        None,
        "Synthetic rubber (crude oil + mixed)",
        "(not modelled)",
        True,
        False,
        False,
        False,
    ),
    # tv
    (
        "tv",
        "Plastic / chemical products (back panel, housing: ABS/PMMA)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Steel products (chassis, frame)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Aluminium products (frame)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Flat glass (LCD/OLED display substrate)",
        "Flat glass",
        None,
        "Silica sand + ITO / OLED substrate minerals",
        "glass-glass / glass-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (wiring, PCB traces)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Metallic minerals (rare earth phosphors, indium ITO)",
        "Metallic minerals",
        None,
        "Rare earth minerals + Indium ore",
        "(not modelled — NACE B mining)",
        True,
        False,
        False,
        False,
    ),
    # phone
    (
        "phone",
        "Aluminium products (chassis, frame)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Flat glass (aluminosilicate screen)",
        "Flat glass",
        None,
        "Silica sand + Aluminium oxide + Boron",
        "glass-glass / glass-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (PCB, wiring)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Plastic / chemical products (back cover, connectors)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Metallic minerals (Li battery, cobalt, gold, silver, tantalum, rare earths)",
        "Metallic minerals",
        None,
        "Cobalt ore + Lithium brine + Gold / Silver / Tantalum ore + Rare earth minerals",
        "(not modelled — NACE B mining)",
        True,
        False,
        False,
        False,
    ),
    note(
        "Entire Li battery mass included in the phone product variable — not tracked separately"
    ),
    # fridge
    (
        "fridge",
        "Steel products (body, liner, shelf supports)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Plastic / chemical products (PU foam insulation, interior liner, door seal)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (compressor, evaporator coil)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Aluminium products (condenser coil, decorative trim)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Paper products (shelf liner)",
        "Paper / paperboard",
        "Pulp",
        "Roundwood (A02 forestry)",
        "paper-tech / paper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Other materials (HFC-134a refrigerant, rubber seals, resin gaskets)",
        "Other materials",
        None,
        "Crude oil + HF (from fluorspar) + mixed",
        "(not modelled)",
        True,
        False,
        False,
        False,
    ),
    # dishwasher
    (
        "dishwasher",
        "Steel products (body, frame, stainless inner tub)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Plastic / chemical products (door liner, spray arms, baskets: PP/ABS)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Asphalt (anti-vibration mastic, sound-deadening coating)",
        "Asphalt",
        None,
        "Crude oil",
        "asphalt-tech / asphalt-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Sawnwood (cabinet structure / packing)",
        "Sawnwood",
        None,
        "Roundwood (A02 forestry)",
        "timber-tech / timber-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (motor windings, heating element)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Aluminium products (wash pump housing, trim)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "OIS products (ceramic spray arm bearings, valve seats)",
        "OIS",
        None,
        "Clay + Alumina",
        "ois-tech / ois-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Other materials (zinc die-castings, brass fittings, rubber seals)",
        "Other materials",
        None,
        "Mixed",
        "(not modelled)",
        True,
        False,
        False,
        False,
    ),
    note(
        "Asphalt entry = rubber-bitumen damping mastic; Processor zeroes dishwasher asphalt so it flows to 'other' at EOL"
    ),
    # wmachine
    (
        "wmachine",
        "Steel products (drum, outer tub, frame)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Plastic / chemical products (tub, door, control panel, rubber: seal, hoses)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Aluminium products (motor frame)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Sawnwood (counterweight support / internal bracing)",
        "Sawnwood",
        None,
        "Roundwood (A02 forestry)",
        "timber-tech / timber-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (motor windings)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Other materials (balance weights, drum bearings, brass fittings, paint)",
        "Other materials",
        None,
        "Mixed",
        "(not modelled)",
        True,
        False,
        False,
        False,
    ),
    # freezer
    (
        "freezer",
        "Steel products (body, liner, shelf supports)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Plastic / chemical products (PU foam insulation, interior liner, door seal)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Copper products (compressor, evaporator coil)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Aluminium products (condenser coil, trim)",
        "Aluminium ingot",
        None,
        "Bauxite",
        "aluminium-prim / aluminium-sec",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Other materials (HFC-134a refrigerant, rubber seals, resin gaskets)",
        "Other materials",
        None,
        "Crude oil + HF (from fluorspar) + mixed",
        "(not modelled)",
        True,
        False,
        False,
        False,
    ),
]


# ── INFRASTRUCTURE ────────────────────────────────────────────────────────────
def CEM_ROW(comp):
    return (
        None,
        f"Concrete ({comp})",
        "Concrete",
        "Cement",
        "Limestone + Clay + Gypsum",
        "cement-dry-kiln / cement-wet-kiln / cement-geopolym / cement-sec-post-consumer",
        False,
        True,
        False,
        False,
    )


infrastructure = [
    (
        "road",
        "Asphalt (pavement surface + binder courses, 840 t/km — 5 cm wearing course × 2.4 t/m³)",
        "Asphalt",
        None,
        "Crude oil",
        "asphalt-tech / asphalt-sec",
        True,
        False,
        False,
        True,
    ),
    CEM_ROW(
        "base layer, 0 t/km in model for road resurfacing; cement applies to full reconstruction only"
    ),
    (
        None,
        None,
        None,
        "Non-metallic minerals",
        "Sand + Gravel + Crushed stone (not in road model — NACE B08 out of scope)",
        "(not modelled — NACE B08 mining)",
        False,
        True,
        False,
        False,
    ),
    note(
        "EUCalc D3.1 source had only concrete-and-inert; asphalt added via Option C in processor; aggregate dropped as NACE B08 out of scope"
    ),
    (
        "rail",
        "Steel products (rail, ⚠ 0.06 t/km in model — likely unit error; UIC60 rail ≈ 120 t/km)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-hisarna / steel-hydrog-DRI / steel-scrap-EAF",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Copper products (overhead catenary wire for electrified lines)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        False,
    ),
    CEM_ROW("concrete sleepers, 0.12 fraction of concrete-and-inert"),
    (
        None,
        None,
        None,
        "Non-metallic minerals",
        "Crushed stone (ballast aggregate, 88% of concrete-and-inert)",
        "(not modelled — NACE B08 mining)",
        False,
        True,
        False,
        False,
    ),
    (
        "trolley-cables",
        "Copper products (contact wire, feeder cable)",
        "Refined copper",
        None,
        "Copper ore",
        "copper-tech / copper-sec",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Steel products (support structures, messenger wire)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-scrap-EAF",
        True,
        False,
        False,
        False,
    ),
    (
        None,
        "Plastic / chemical products (insulation, conduit)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
    (
        "new-dhg-pipe",
        "Steel products (carrier pipe)",
        "Crude steel",
        None,
        "Iron ore + Coking coal",
        "steel-BF-BOF / steel-hisarna / steel-hydrog-DRI / steel-scrap-EAF",
        True,
        False,
        False,
        True,
    ),
    (
        None,
        "Plastic / chemical products (insulation jacket, PUR foam, HDPE outer casing)",
        "Polymers / resins",
        None,
        "Crude oil + Natural gas",
        "chem-chem-tech / chem-sec",
        True,
        False,
        False,
        False,
    ),
]

# ── FERTILIZER ────────────────────────────────────────────────────────────────
fertilizer = [
    (
        "fertilizer",
        "Nitrogen fertilizer (urea, ammonium nitrate, ammonium sulphate)",
        "Nitrogen fertilizer",
        "Ammonia",
        "Natural gas (H source) + Air (N₂)",
        "ammonia-tech (industry preprocessing); full tech split in ammonia sector module",
        False,
        True,
        False,
        True,
    ),
    note(
        "Fertilizer maps 1:1 to ammonia in the model; phosphate (from phosphate rock) and potash (KCl) fertilizers not modelled — both are Non-metallic minerals (NACE B08) falling outside current scope"
    ),
]

# ── BUILD ─────────────────────────────────────────────────────────────────────
wb = openpyxl.Workbook()
wb.remove(wb.active)

write_readme(wb, "README")
write_map(wb, "Material name mapping", mapping)
write_main(wb, "Buildings material chain", buildings)
write_main(wb, "Vehicles material chain", vehicles)
write_main(wb, "Batteries material chain", batteries)
write_main(wb, "Packaging and paper", packaging)
write_main(wb, "Appliances material chain", appliances)
write_main(wb, "Infrastructure material chain", infrastructure)
write_main(wb, "Fertilizer material chain", fertilizer)

path = "transition_compass_model/_database/pre_processing/industry/eu/data/Literature/raw/material_chain.xlsx"
wb.save(path)

wb2 = openpyxl.load_workbook(path)
print("Sheets:", wb2.sheetnames)
for sh in wb2.sheetnames:
    ws = wb2[sh]
    print(f"  {sh}: {ws.max_row} rows x {ws.max_column} cols")
print("Done.")
