for ad_pref_tag in GPT4_Preference GPT4_DS Puyu20b_DS Puyu20b_Preference 
do 
    AD_PREF_TAG=$ad_pref_tag bash scripts/eval_intern_pm_0401_v1.sh
done
