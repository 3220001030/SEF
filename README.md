# SEF

```Python
import delimited "/Users/gurumakaza/Documents/data/full_qap_results_interpolated.csv", clear
encode city, gen(city_id)
encode type, gen(type_id)
keep if type == "内资"
duplicates drop city_id year, force
xtset city_id year
local depvars esp_qol qol_lod lod_rds qol_rds ef_rds rds_wi ef_es wi_es es_esp

cap erase qap_panel_fe.doc
foreach v of local depvars {
    xtreg `v' i.year, fe vce(robust)
    outreg2 using qap_panel_fe.doc, append ctitle(`v') alpha(0.001, 0.01, 0.05) bdec(3) tdec(3) addstat(R-squared, `e(r2)', N, e(N))
}
shellout using `"qap_panel_fe.doc"'
seeout using "qap_panel_fe.txt"
shellout using `"qap_panel_fe.doc"'
```
