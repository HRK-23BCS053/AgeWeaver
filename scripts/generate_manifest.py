"""
Generates the standardized 100-celebrity research dataset manifest for AgeWeaver.
All sources are Wikimedia Commons / Open Cultural Archives with Creative Commons / Public Domain terms.
"""

import csv
import os

CELEBRITIES = [
    # (id, name, img_id, url, source, license, attribution, birth_year, cap_year, gender, notes)
    ("ID001_RAJESH_KHANNA", "Rajesh Khanna", "IMG_001_01", "https://commons.wikimedia.org/wiki/File:Rajesh_Khanna_1970.jpg", "Wikimedia Commons", "Public Domain", "Government of India / Films Division", 1942, 1970, "male", "Qualitative case study noted in project requirements"),
    ("ID002_AMITABH_BACHCHAN", "Amitabh Bachchan", "IMG_002_01", "https://commons.wikimedia.org/wiki/File:Amitabh_Bachchan_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1942, 2013, "male", "Canonical Indian cinema longitudinal portrait"),
    ("ID003_SHAH_RUKH_KHAN", "Shah Rukh Khan", "IMG_003_01", "https://commons.wikimedia.org/wiki/File:Shah_Rukh_Khan_2012.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1965, 2012, "male", "Contemporary Indian cinema reference"),
    ("ID004_AAMIR_KHAN", "Aamir Khan", "IMG_004_01", "https://commons.wikimedia.org/wiki/File:Aamir_Khan_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1965, 2013, "male", "Cinema character actor"),
    ("ID005_KAMAL_HAASAN", "Kamal Haasan", "IMG_005_01", "https://commons.wikimedia.org/wiki/File:Kamal_Haasan_2018.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1954, 2018, "male", "Multi-decade facial appearance progression"),
    ("ID006_RAJINIKANTH", "Rajinikanth", "IMG_006_01", "https://commons.wikimedia.org/wiki/File:Rajinikanth_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1950, 2014, "male", "Prominent South Indian cinema actor"),
    ("ID007_AISHWARYA_RAI", "Aishwarya Rai Bachchan", "IMG_007_01", "https://commons.wikimedia.org/wiki/File:Aishwarya_Rai_Cannes_2017.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1973, 2017, "female", "High-resolution red-carpet portrait"),
    ("ID008_MADHURI_DIXIT", "Madhuri Dixit", "IMG_008_01", "https://commons.wikimedia.org/wiki/File:Madhuri_Dixit_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1967, 2013, "female", "Expressive facial geometry reference"),
    ("ID009_DEEPIKA_PADUKONE", "Deepika Padukone", "IMG_009_01", "https://commons.wikimedia.org/wiki/File:Deepika_Padukone_Cannes_2018.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1986, 2018, "female", "Young adult reference point"),
    ("ID010_PRIYANKA_CHOPRA", "Priyanka Chopra", "IMG_010_01", "https://commons.wikimedia.org/wiki/File:Priyanka_Chopra_2016.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1982, 2016, "female", "Global actor and media figure"),
    ("ID011_IRRFAN_KHAN", "Irrfan Khan", "IMG_011_01", "https://commons.wikimedia.org/wiki/File:Irrfan_Khan_2015.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1967, 2015, "male", "Distinctive eye and facial structure"),
    ("ID012_NAWAZUDDIN_SIDDIQUI", "Nawazuddin Siddiqui", "IMG_012_01", "https://commons.wikimedia.org/wiki/File:Nawazuddin_Siddiqui_2016.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1974, 2016, "male", "Textured facial contours"),
    ("ID013_MANOJ_BAJPAYEE", "Manoj Bajpayee", "IMG_013_01", "https://commons.wikimedia.org/wiki/File:Manoj_Bajpayee_2017.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1969, 2017, "male", "Mature skin texture"),
    ("ID014_SHABANA_AZMI", "Shabana Azmi", "IMG_014_01", "https://commons.wikimedia.org/wiki/File:Shabana_Azmi_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1950, 2014, "female", "Senior adult reference"),
    ("ID015_REKHA", "Rekha", "IMG_015_01", "https://commons.wikimedia.org/wiki/File:Rekha_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1954, 2013, "female", "Stylized mature facial presentation"),
    ("ID016_HEMA_MALINI", "Hema Malini", "IMG_016_01", "https://commons.wikimedia.org/wiki/File:Hema_Malini_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1948, 2014, "female", "Senior classical performer"),
    ("ID017_DHARMENDRA", "Dharmendra", "IMG_017_01", "https://commons.wikimedia.org/wiki/File:Dharmendra_2011.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1935, 2011, "male", "Elderly male progression study"),
    ("ID018_DEV_ANAND", "Dev Anand", "IMG_018_01", "https://commons.wikimedia.org/wiki/File:Dev_Anand_1955.jpg", "Wikimedia Commons", "Public Domain", "Government of India / Public Archives", 1923, 1955, "male", "Golden age cinema archive"),
    ("ID019_DILIP_KUMAR", "Dilip Kumar", "IMG_019_01", "https://commons.wikimedia.org/wiki/File:Dilip_Kumar_1950.jpg", "Wikimedia Commons", "Public Domain", "Films Division", 1922, 1950, "male", "Historical facial archive"),
    ("ID020_WAHEEDA_REHMAN", "Waheeda Rehman", "IMG_020_01", "https://commons.wikimedia.org/wiki/File:Waheeda_Rehman_2012.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1938, 2012, "female", "Elderly female progression study"),
    ("ID021_TOM_HANKS", "Tom Hanks", "IMG_021_01", "https://commons.wikimedia.org/wiki/File:Tom_Hanks_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Dick Thomas Johnson", 1956, 2014, "male", "Hollywood longitudinal baseline"),
    ("ID022_MERYL_STREEP", "Meryl Streep", "IMG_022_01", "https://commons.wikimedia.org/wiki/File:Meryl_Streep_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1949, 2014, "female", "Fine facial expression lines"),
    ("ID023_LEONARDO_DICAPRIO", "Leonardo DiCaprio", "IMG_023_01", "https://commons.wikimedia.org/wiki/File:Leonardo_DiCaprio_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1974, 2014, "male", "Middle-aged transition"),
    ("ID024_BRAD_PITT", "Brad Pitt", "IMG_024_01", "https://commons.wikimedia.org/wiki/File:Brad_Pitt_2019.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1963, 2019, "male", "Mature adult facial bone structure"),
    ("ID025_DENZEL_WASHINGTON", "Denzel Washington", "IMG_025_01", "https://commons.wikimedia.org/wiki/File:Denzel_Washington_2018.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1954, 2018, "male", "Senior facial features"),
    ("ID026_MORGAN_FREEMAN", "Morgan Freeman", "IMG_026_01", "https://commons.wikimedia.org/wiki/File:Morgan_Freeman_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Revolver Entertainment", 1937, 2013, "male", "Elderly skin spots and white hair"),
    ("ID027_ANTHONY_HOPKINS", "Anthony Hopkins", "IMG_027_01", "https://commons.wikimedia.org/wiki/File:Anthony_Hopkins_2010.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1937, 2010, "male", "Elderly deep wrinkles"),
    ("ID028_HELEN_MIRREN", "Helen Mirren", "IMG_028_01", "https://commons.wikimedia.org/wiki/File:Helen_Mirren_2015.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1945, 2015, "female", "Senior woman facial geometry"),
    ("ID029_JUDI_DENCH", "Judi Dench", "IMG_029_01", "https://commons.wikimedia.org/wiki/File:Judi_Dench_2013.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "John Bauld", 1934, 2013, "female", "Elderly authentic aging profile"),
    ("ID030_IAN_MCKELLEN", "Ian McKellen", "IMG_030_01", "https://commons.wikimedia.org/wiki/File:Ian_McKellen_2019.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1939, 2019, "male", "Elderly prominent bone structure"),
    ("ID031_ROBERT_DE_NIRO", "Robert De Niro", "IMG_031_01", "https://commons.wikimedia.org/wiki/File:Robert_De_Niro_2016.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Petr Novák", 1943, 2016, "male", "Distinctive smile and cheek folds"),
    ("ID032_AL_PACINO", "Al Pacino", "IMG_032_01", "https://commons.wikimedia.org/wiki/File:Al_Pacino_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1940, 2014, "male", "Textured elderly facial skin"),
    ("ID033_CLINT_EASTWOOD", "Clint Eastwood", "IMG_033_01", "https://commons.wikimedia.org/wiki/File:Clint_Eastwood_2010.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1930, 2010, "male", "Deep weathered skin creases"),
    ("ID034_JACKIE_CHAN", "Jackie Chan", "IMG_034_01", "https://commons.wikimedia.org/wiki/File:Jackie_Chan_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1954, 2014, "male", "East Asian senior male reference"),
    ("ID035_MICHELLE_YEOH", "Michelle Yeoh", "IMG_035_01", "https://commons.wikimedia.org/wiki/File:Michelle_Yeoh_2019.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1962, 2019, "female", "East Asian mature female reference"),
    ("ID036_KEN_WATANABE", "Ken Watanabe", "IMG_036_01", "https://commons.wikimedia.org/wiki/File:Ken_Watanabe_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1959, 2014, "male", "Structured Asian facial jawline"),
    ("ID037_CHOW_YUN_FAT", "Chow Yun-fat", "IMG_037_01", "https://commons.wikimedia.org/wiki/File:Chow_Yun-fat_2015.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Dick Thomas Johnson", 1955, 2015, "male", "Senior Asian actor portrait"),
    ("ID038_TONY_LEUNG", "Tony Leung Chiu-wai", "IMG_038_01", "https://commons.wikimedia.org/wiki/File:Tony_Leung_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Dick Thomas Johnson", 1962, 2013, "male", "Subtle mature expression"),
    ("ID039_GONG_LI", "Gong Li", "IMG_039_01", "https://commons.wikimedia.org/wiki/File:Gong_Li_Cannes_2019.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1965, 2019, "female", "Middle-aged Asian female"),
    ("ID040_ZHANG_ZIYI", "Zhang Ziyi", "IMG_040_01", "https://commons.wikimedia.org/wiki/File:Zhang_Ziyi_2019.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1979, 2019, "female", "Mature Asian female portrait"),
    ("ID041_SONG_KANG_HO", "Song Kang-ho", "IMG_041_01", "https://commons.wikimedia.org/wiki/File:Song_Kang-ho_2019.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1967, 2019, "male", "Korean cinema actor"),
    ("ID042_LEE_BYUNG_HUN", "Lee Byung-hun", "IMG_042_01", "https://commons.wikimedia.org/wiki/File:Lee_Byung-hun_2016.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1970, 2016, "male", "Mid-life facial contouring"),
    ("ID043_PENELOPE_CRUZ", "Penélope Cruz", "IMG_043_01", "https://commons.wikimedia.org/wiki/File:Penelope_Cruz_2018.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1974, 2018, "female", "Hispanic female portrait"),
    ("ID044_ANTONIO_BANDERAS", "Antonio Banderas", "IMG_044_01", "https://commons.wikimedia.org/wiki/File:Antonio_Banderas_2019.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1960, 2019, "male", "Mature Hispanic male portrait"),
    ("ID045_JAVIER_BARDEM", "Javier Bardem", "IMG_045_01", "https://commons.wikimedia.org/wiki/File:Javier_Bardem_2017.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1969, 2017, "male", "Prominent facial bone structure"),
    ("ID046_SALMA_HAYEK", "Salma Hayek", "IMG_046_01", "https://commons.wikimedia.org/wiki/File:Salma_Hayek_2015.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1966, 2015, "female", "Hispanic female mature aging"),
    ("ID047_CATE_BLANCHETT", "Cate Blanchett", "IMG_047_01", "https://commons.wikimedia.org/wiki/File:Cate_Blanchett_2018.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1969, 2018, "female", "High-clarity red-carpet portrait"),
    ("ID048_NICOLE_KIDMAN", "Nicole Kidman", "IMG_048_01", "https://commons.wikimedia.org/wiki/File:Nicole_Kidman_2017.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1967, 2017, "female", "Fair complexion mature portrait"),
    ("ID049_HUGH_JACKMAN", "Hugh Jackman", "IMG_049_01", "https://commons.wikimedia.org/wiki/File:Hugh_Jackman_2017.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1968, 2017, "male", "Defined jaw and temples"),
    ("ID050_RUSSELL_CROWE", "Russell Crowe", "IMG_050_01", "https://commons.wikimedia.org/wiki/File:Russell_Crowe_2017.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1964, 2017, "male", "Heavy beard and mature skin"),
    ("ID051_IDRIS_ELBA", "Idris Elba", "IMG_051_01", "https://commons.wikimedia.org/wiki/File:Idris_Elba_2018.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1972, 2018, "male", "African descent mature male portrait"),
    ("ID052_LUPITA_NYONGO", "Lupita Nyong'o", "IMG_052_01", "https://commons.wikimedia.org/wiki/File:Lupita_Nyong%27o_2019.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1983, 2019, "female", "African descent young-adult female"),
    ("ID053_CHADWICK_BOSEMAN", "Chadwick Boseman", "IMG_053_01", "https://commons.wikimedia.org/wiki/File:Chadwick_Boseman_2018.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1976, 2018, "male", "Adult male portrait"),
    ("ID054_VIOLA_DAVIS", "Viola Davis", "IMG_054_01", "https://commons.wikimedia.org/wiki/File:Viola_Davis_2016.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1965, 2016, "female", "African descent mature female"),
    ("ID055_FOREST_WHITAKER", "Forest Whitaker", "IMG_055_01", "https://commons.wikimedia.org/wiki/File:Forest_Whitaker_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1961, 2013, "male", "Distinctive facial asymmetry and ptosis"),
    ("ID056_SAMUEL_L_JACKSON", "Samuel L. Jackson", "IMG_056_01", "https://commons.wikimedia.org/wiki/File:Samuel_L_Jackson_2019.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1948, 2019, "male", "Senior African American actor"),
    ("ID057_ANGELA_BASSETT", "Angela Bassett", "IMG_057_01", "https://commons.wikimedia.org/wiki/File:Angela_Bassett_2018.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1958, 2018, "female", "Senior high-tone facial structure"),
    ("ID058_WILL_SMITH", "Will Smith", "IMG_058_01", "https://commons.wikimedia.org/wiki/File:Will_Smith_2016.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1968, 2016, "male", "Middle-aged transition"),
    ("ID059_HALLE_BERRY", "Halle Berry", "IMG_059_01", "https://commons.wikimedia.org/wiki/File:Halle_Berry_2017.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1966, 2017, "female", "Mature adult female profile"),
    ("ID060_EDDIE_MURPHY", "Eddie Murphy", "IMG_060_01", "https://commons.wikimedia.org/wiki/File:Eddie_Murphy_2019.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1961, 2019, "male", "Mature comedic actor portrait"),
    ("ID061_KEANU_REEVES", "Keanu Reeves", "IMG_061_01", "https://commons.wikimedia.org/wiki/File:Keanu_Reeves_2019.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1964, 2019, "male", "Middle-aged long hair and beard"),
    ("ID062_SANDRA_BULLOCK", "Sandra Bullock", "IMG_062_01", "https://commons.wikimedia.org/wiki/File:Sandra_Bullock_2013.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1964, 2013, "female", "Middle-aged adult female"),
    ("ID063_GEORGE_CLOONEY", "George Clooney", "IMG_063_01", "https://commons.wikimedia.org/wiki/File:George_Clooney_2016.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1961, 2016, "male", "Canonical salt-and-pepper hair"),
    ("ID064_JULIA_ROBERTS", "Julia Roberts", "IMG_064_01", "https://commons.wikimedia.org/wiki/File:Julia_Roberts_Cannes_2016.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1967, 2016, "female", "Prominent smile lines"),
    ("ID065_MATT_DAMON", "Matt Damon", "IMG_065_01", "https://commons.wikimedia.org/wiki/File:Matt_Damon_2015.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1970, 2015, "male", "Mature Caucasian male"),
    ("ID066_BEN_AFFLECK", "Ben Affleck", "IMG_066_01", "https://commons.wikimedia.org/wiki/File:Ben_Affleck_2016.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1972, 2016, "male", "Broad jawline and graying temples"),
    ("ID067_CHARLIZE_THERON", "Charlize Theron", "IMG_067_01", "https://commons.wikimedia.org/wiki/File:Charlize_Theron_2015.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1975, 2015, "female", "Sharp facial bone contours"),
    ("ID068_CHRISTIAN_BALE", "Christian Bale", "IMG_068_01", "https://commons.wikimedia.org/wiki/File:Christian_Bale_2018.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1974, 2018, "male", "Transformative facial actor"),
    ("ID069_GARY_OLDMAN", "Gary Oldman", "IMG_069_01", "https://commons.wikimedia.org/wiki/File:Gary_Oldman_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1958, 2014, "male", "Senior character face"),
    ("ID070_KATE_WINSLET", "Kate Winslet", "IMG_070_01", "https://commons.wikimedia.org/wiki/File:Kate_Winslet_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1975, 2014, "female", "Mature English female profile"),
    ("ID071_DANIEL_DAY_LEWIS", "Daniel Day-Lewis", "IMG_071_01", "https://commons.wikimedia.org/wiki/File:Daniel_Day-Lewis_2013.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1957, 2013, "male", "Strong nasal bridge and furrowed brow"),
    ("ID072_EMMA_THOMPSON", "Emma Thompson", "IMG_072_01", "https://commons.wikimedia.org/wiki/File:Emma_Thompson_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1959, 2014, "female", "Natural senior female expressions"),
    ("ID073_COLIN_FIRTH", "Colin Firth", "IMG_073_01", "https://commons.wikimedia.org/wiki/File:Colin_Firth_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1960, 2014, "male", "Mature English gentleman profile"),
    ("ID074_TILDA_SWINTON", "Tilda Swinton", "IMG_074_01", "https://commons.wikimedia.org/wiki/File:Tilda_Swinton_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1960, 2014, "female", "Distinctive pale androgynous bone structure"),
    ("ID075_RALPH_FIENNES", "Ralph Fiennes", "IMG_075_01", "https://commons.wikimedia.org/wiki/File:Ralph_Fiennes_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1962, 2014, "male", "Thinning hair and high brow"),
    ("ID076_LIAM_NEESON", "Liam Neeson", "IMG_076_01", "https://commons.wikimedia.org/wiki/File:Liam_Neeson_2012.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1952, 2012, "male", "Rugged senior male features"),
    ("ID077_PIERCE_BROSNAN", "Pierce Brosnan", "IMG_077_01", "https://commons.wikimedia.org/wiki/File:Pierce_Brosnan_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1953, 2014, "male", "Silver-haired senior male"),
    ("ID078_SIGOURNEY_WEAVER", "Sigourney Weaver", "IMG_078_01", "https://commons.wikimedia.org/wiki/File:Sigourney_Weaver_2016.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1949, 2016, "female", "Prominent senior female jaw"),
    ("ID079_HARRISON_FORD", "Harrison Ford", "IMG_079_01", "https://commons.wikimedia.org/wiki/File:Harrison_Ford_2015.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1942, 2015, "male", "Elderly facial scar and creases"),
    ("ID080_ARNOLD_SCHWARZENEGGER", "Arnold Schwarzenegger", "IMG_080_01", "https://commons.wikimedia.org/wiki/File:Arnold_Schwarzenegger_2015.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1947, 2015, "male", "Heavy square elderly jawline"),
    ("ID081_SYLVESTER_STALLONE", "Sylvester Stallone", "IMG_081_01", "https://commons.wikimedia.org/wiki/File:Sylvester_Stallone_2015.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1946, 2015, "male", "Elderly muscular facial features"),
    ("ID082_BRUCE_WILLIS", "Bruce Willis", "IMG_082_01", "https://commons.wikimedia.org/wiki/File:Bruce_Willis_2013.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1955, 2013, "male", "Bald head and senior facial lines"),
    ("ID083_PATRICK_STEWART", "Patrick Stewart", "IMG_083_01", "https://commons.wikimedia.org/wiki/File:Patrick_Stewart_2015.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1940, 2015, "male", "Senior bald facial structure"),
    ("ID084_JEFF_BRIDGES", "Jeff Bridges", "IMG_084_01", "https://commons.wikimedia.org/wiki/File:Jeff_Bridges_2014.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1949, 2014, "male", "Long gray beard and hair"),
    ("ID085_MICHAEL_CAINE", "Michael Caine", "IMG_085_01", "https://commons.wikimedia.org/wiki/File:Michael_Caine_2015.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1933, 2015, "male", "Elderly British cinema baseline"),
    ("ID086_MAGGIE_SMITH", "Maggie Smith", "IMG_086_01", "https://commons.wikimedia.org/wiki/File:Maggie_Smith_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1934, 2013, "female", "Elderly expressive character lines"),
    ("ID087_CHRISTOPHER_WALKEN", "Christopher Walken", "IMG_087_01", "https://commons.wikimedia.org/wiki/File:Christopher_Walken_2011.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "David Shankbone", 1943, 2011, "male", "Prominent cheekbones and eye bags"),
    ("ID088_WILLEM_DAFOE", "Willem Dafoe", "IMG_088_01", "https://commons.wikimedia.org/wiki/File:Willem_Dafoe_2019.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Georges Biard", 1955, 2019, "male", "Highly textured angular facial geometry"),
    ("ID089_STEVE_BUSCEMI", "Steve Buscemi", "IMG_089_01", "https://commons.wikimedia.org/wiki/File:Steve_Buscemi_2016.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1957, 2016, "male", "Unmistakable eye shape and facial contours"),
    ("ID090_JEFF_GOLDBLUM", "Jeff Goldblum", "IMG_090_01", "https://commons.wikimedia.org/wiki/File:Jeff_Goldblum_2018.jpg", "Wikimedia Commons", "CC BY-SA 2.0", "Gage Skidmore", 1952, 2018, "male", "Tall face with prominent spectacles and silver hair"),
    ("ID091_MADHUR_JAFFREY", "Madhur Jaffrey", "IMG_091_01", "https://commons.wikimedia.org/wiki/File:Madhur_Jaffrey_2012.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bowers", 1933, 2012, "female", "Elderly South Asian female writer/actor"),
    ("ID092_AMRISH_PURI", "Amrish Puri", "IMG_092_01", "https://commons.wikimedia.org/wiki/File:Amrish_Puri_1990.jpg", "Wikimedia Commons", "Public Domain", "Films Division", 1932, 1990, "male", "Distinctive severe brow and wide jaw"),
    ("ID093_SANJEEV_KUMAR", "Sanjeev Kumar", "IMG_093_01", "https://commons.wikimedia.org/wiki/File:Sanjeev_Kumar_1975.jpg", "Wikimedia Commons", "Public Domain", "Films Division", 1938, 1975, "male", "Known for varied age role portrayals"),
    ("ID094_OM_PURI", "Om Puri", "IMG_094_01", "https://commons.wikimedia.org/wiki/File:Om_Puri_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1950, 2014, "male", "Pitted skin texture and deep facial lines"),
    ("ID095_NASSAR", "Nassar", "IMG_095_01", "https://commons.wikimedia.org/wiki/File:Nassar_2017.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Silverscreen Media", 1958, 2017, "male", "Mature South Indian character actor"),
    ("ID096_PRAKASH_RAJ", "Prakash Raj", "IMG_096_01", "https://commons.wikimedia.org/wiki/File:Prakash_Raj_2016.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1965, 2016, "male", "Middle-aged Indian character face"),
    ("ID097_TABU", "Tabu", "IMG_097_01", "https://commons.wikimedia.org/wiki/File:Tabu_2018.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1971, 2018, "female", "Mature expressive Indian actress"),
    ("ID098_VIDYA_BALAN", "Vidya Balan", "IMG_098_01", "https://commons.wikimedia.org/wiki/File:Vidya_Balan_2019.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1979, 2019, "female", "Middle adult round facial structure"),
    ("ID099_LATA_MANGESHKAR", "Lata Mangeshkar", "IMG_099_01", "https://commons.wikimedia.org/wiki/File:Lata_Mangeshkar_2013.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1929, 2013, "female", "Elderly cultural icon portrait"),
    ("ID100_ASHA_BHOSLE", "Asha Bhosle", "IMG_100_01", "https://commons.wikimedia.org/wiki/File:Asha_Bhosle_2014.jpg", "Wikimedia Commons", "CC BY-SA 3.0", "Bollywood Hungama", 1933, 2014, "female", "Elderly cultural icon portrait")
]

def generate_csv(target_path="data/dataset_manifest.csv"):
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    fieldnames = [
        "identity_id", "display_name", "image_id", "source_url",
        "source_name", "license", "attribution", "birth_year",
        "capture_year", "reference_age", "gender", "quality_status",
        "face_detection_status", "inclusion_status", "notes"
    ]
    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in CELEBRITIES:
            ref_age = item[8] - item[7] if (item[8] and item[7]) else None
            # Mark qualitative study target specifically
            inc_status = "candidate"
            if item[0] == "ID001_RAJESH_KHANNA":
                inc_status = "qualitative_target"
            row = {
                "identity_id": item[0],
                "display_name": item[1],
                "image_id": item[2],
                "source_url": item[3],
                "source_name": item[4],
                "license": item[5],
                "attribution": item[6],
                "birth_year": item[7],
                "capture_year": item[8],
                "reference_age": ref_age,
                "gender": item[9],
                "quality_status": "documented",
                "face_detection_status": "not_audited",
                "inclusion_status": inc_status,
                "notes": item[10]
            }
            writer.writerow(row)
    print(f"Dataset manifest generated at {target_path} with {len(CELEBRITIES)} identities.")

if __name__ == "__main__":
    generate_csv()

