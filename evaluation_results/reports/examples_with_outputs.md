## Concrete Examples with Model Outputs

Let's examine specific cases to see what each model actually returned. Results categorized as:
- **✓✓ Source**: The original source case
- **✓ Cited**: Cases cited in the source case's opinion (relevant)
- **✗ Unrelated**: Cases not relevant to the query

**Note**: P@5 = 0.552 is an AVERAGE across all queries. Individual queries vary - some retrieve more cited cases than others.

### Example: Trump v. CASA, Inc.

#### Extractive Query (Query ID: 346)
> "TRUMP, PRESIDENT OF THE UNITED STATES, et al , et al Plaintiffs (respondents here)—individuals, organizations, and States—fled three separate suits to enjoin the implementation and enforcement of Pres..."
> 
> *Source: Trump v. CASA, Inc.*

#### Generative Query (Query ID: 1001)
> "Can the President issue an executive order to redefine birthright citizenship under the 14th Amendment?..."

#### Model Results for Extractive Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 1 | Trump v. CASA, Inc. | ✓✓ Source |
| 2 | 112 | Trump v. Anderson | ✗ Unrelated |
| 3 | 313 | Department of Homeland Security v. Regents of Univ. of Cal. | ✗ Unrelated |
| 4 | 173 | Biden v. Texas | ✗ Unrelated |
| 5 | 300 | Trump v. Vance | ✗ Unrelated |
| 6 | 289 | Trump v. New York | ✗ Unrelated |
| 7 | 199 | Garland v. Gonzalez | ✗ Unrelated |
| 8 | 240 | Johnson v. Guzman Chavez | ✗ Unrelated |
| 9 | 299 | Trump v. Mazars USA, LLP | ✗ Unrelated |
| 10 | 63 | Trump v. United States | ✓ Cited |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 163 | New York v. New Jersey | ✗ Unrelated |
| 2 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 3 | 230 | Biden v. Missouri | ✗ Unrelated |
| 4 | 243 | Lombardo v. St. Louis | ✗ Unrelated |
| 5 | 124 | Moore v. Harper | ✗ Unrelated |
| 6 | 334 | Atlantic Richfield Co. v. Christian | ✗ Unrelated |
| 7 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 8 | 308 | Agency for Int’l Development v. Alliance for Open Society | ✗ Unrelated |
| 9 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 10 | 15 | Fuld v. Palestine Liberation Organization | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 2 | 275 | Tandon v. Newsom | ✓ Cited |
| 3 | 40 | Department of Education v. California | ✗ Unrelated |
| 4 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 5 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 6 | 295 | Tanzin v. Tanvir | ✗ Unrelated |
| 7 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 8 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 9 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 10 | 230 | Biden v. Missouri | ✗ Unrelated |

#### Model Results for Generative Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 1 | Trump v. CASA, Inc. | ✓✓ Source |
| 2 | 313 | Department of Homeland Security v. Regents of Univ. of Cal. | ✗ Unrelated |
| 3 | 300 | Trump v. Vance | ✗ Unrelated |
| 4 | 173 | Biden v. Texas | ✗ Unrelated |
| 5 | 112 | Trump v. Anderson | ✗ Unrelated |
| 6 | 125 | United States v. Hansen | ✗ Unrelated |
| 7 | 289 | Trump v. New York | ✗ Unrelated |
| 8 | 284 | Pereida v. Wilkinson | ✗ Unrelated |
| 9 | 199 | Garland v. Gonzalez | ✗ Unrelated |
| 10 | 305 | Chiafalo v. Washington | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 308 | Agency for Int’l Development v. Alliance for Open Society | ✗ Unrelated |
| 2 | 124 | Moore v. Harper | ✗ Unrelated |
| 3 | 243 | Lombardo v. St. Louis | ✗ Unrelated |
| 4 | 11 | Esteras v. United States | ✗ Unrelated |
| 5 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 6 | 230 | Biden v. Missouri | ✗ Unrelated |
| 7 | 163 | New York v. New Jersey | ✗ Unrelated |
| 8 | 161 | Turkiye Halk Bankasi A.S. v. United States | ✗ Unrelated |
| 9 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 10 | 330 | Romag Fasteners, Inc. v. Fossil Group, Inc. | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 275 | Tandon v. Newsom | ✓ Cited |
| 4 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 5 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 6 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 7 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 8 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 9 | 295 | Tanzin v. Tanvir | ✗ Unrelated |
| 10 | 297 | Sharp v. Murphy | ✗ Unrelated |

**Analysis:**
- **sentence-transformer**: ✅ Found source case at rank 1 for BOTH query types
- **legal-bert**: Source found at rank 43 (extractive), rank 45 (generative)
- **harvard-bert**: Source found at rank 73 (extractive), rank 97 (generative)

---

### Example: Kennedy v. Braidwood Management, Inc.

#### Extractive Query (Query ID: 347)
> "KENNEDY, SECRETARY OF HEALTH AND HUMAN SERVICES, et al BRAIDWOOD MANAGEMENT, certiorari to the united states court of appeals for In 1984, the Department of Health and Human Services (HHS) created the..."
> 
> *Source: Kennedy v. Braidwood Management, Inc.*

#### Generative Query (Query ID: 1002)
> "Does the Preventive Services Task Force have the authority to mandate coverage for certain healthcare services without Congressional approval?..."

#### Model Results for Extractive Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 2 | Kennedy v. Braidwood Management, Inc. | ✓✓ Source |
| 2 | 180 | Becerra v. Empire Health Foundation, For Valley Hospital Medical Center | ✗ Unrelated |
| 3 | 192 | American Hospital Assn. v. Becerra | ✗ Unrelated |
| 4 | 327 | Maine Community Health Options v. United States | ✗ Unrelated |
| 5 | 229 | NFIB v. OSHA | ✗ Unrelated |
| 6 | 138 | Health and Hospital Corporation of Marion Cty. v. Talevski | ✗ Unrelated |
| 7 | 134 | United States ex rel. Polansky v. Executive Health Resources, Inc. | ✗ Unrelated |
| 8 | 313 | Department of Homeland Security v. Regents of Univ. of Cal. | ✗ Unrelated |
| 9 | 319 | Banister v. Davis | ✗ Unrelated |
| 10 | 1 | Trump v. CASA, Inc. | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 4 | 15 | Fuld v. Palestine Liberation Organization | ✗ Unrelated |
| 5 | 42 | FDA v. Wages and White Lion Investments, LLC | ✗ Unrelated |
| 6 | 334 | Atlantic Richfield Co. v. Christian | ✗ Unrelated |
| 7 | 174 | West Virginia v. EPA | ✗ Unrelated |
| 8 | 124 | Moore v. Harper | ✗ Unrelated |
| 9 | 253 | United States v. Arthrex, Inc. | ✓ Cited |
| 10 | 163 | New York v. New Jersey | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 66 | Loper Bright Enterprises v. Raimondo | ✓ Cited |
| 2 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 3 | 253 | United States v. Arthrex, Inc. | ✓ Cited |
| 4 | 40 | Department of Education v. California | ✗ Unrelated |
| 5 | 230 | Biden v. Missouri | ✗ Unrelated |
| 6 | 295 | Tanzin v. Tanvir | ✗ Unrelated |
| 7 | 273 | Carr v. Saul | ✗ Unrelated |
| 8 | 316 | Lomax v. Ortiz-Marquez | ✗ Unrelated |
| 9 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 10 | 267 | Edwards v. Vannoy | ✗ Unrelated |

#### Model Results for Generative Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 2 | Kennedy v. Braidwood Management, Inc. | ✓✓ Source |
| 2 | 327 | Maine Community Health Options v. United States | ✗ Unrelated |
| 3 | 180 | Becerra v. Empire Health Foundation, For Valley Hospital Medical Center | ✗ Unrelated |
| 4 | 7 | Medina v. Planned Parenthood South Atlantic | ✗ Unrelated |
| 5 | 192 | American Hospital Assn. v. Becerra | ✗ Unrelated |
| 6 | 254 | California v. Texas | ✗ Unrelated |
| 7 | 229 | NFIB v. OSHA | ✗ Unrelated |
| 8 | 36 | Advocate Christ Medical Center v. Kennedy | ✗ Unrelated |
| 9 | 89 | Becerra v. San Carlos Apache Tribe | ✗ Unrelated |
| 10 | 309 | June Medical Services L. L. C. v. Russo | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 66 | Loper Bright Enterprises v. Raimondo | ✓ Cited |
| 2 | 186 | Marietta Memorial Hospital Employee Health Benefit Plan v. DaVita Inc. | ✗ Unrelated |
| 3 | 188 | United States v. Washington | ✗ Unrelated |
| 4 | 28 | Catholic Charities Bureau, Inc. v. Wisconsin Labor and Industry Review Comm’n. | ✗ Unrelated |
| 5 | 17 | EPA v. Calumet Shreveport Refining, L.L.C. | ✗ Unrelated |
| 6 | 1 | Trump v. CASA, Inc. | ✗ Unrelated |
| 7 | 12 | McLaughlin Chiropractic Associates, Inc. v. McKesson Corp. | ✗ Unrelated |
| 8 | 220 | Wisconsin Legislature v. Wisconsin Elections Commission | ✗ Unrelated |
| 9 | 192 | American Hospital Assn. v. Becerra | ✗ Unrelated |
| 10 | 31 | Seven County Infrastructure Coalition v. Eagle County | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 4 | 66 | Loper Bright Enterprises v. Raimondo | ✓ Cited |
| 5 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 6 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 7 | 253 | United States v. Arthrex, Inc. | ✓ Cited |
| 8 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 9 | 297 | Sharp v. Murphy | ✗ Unrelated |
| 10 | 295 | Tanzin v. Tanvir | ✗ Unrelated |

**Analysis:**
- **sentence-transformer**: ✅ Found source case at rank 1 for BOTH query types
- **legal-bert**: Source found at rank 71 (extractive)
- **harvard-bert**: ❌ Failed to find source case in top-100 for both queries

---

### Example: Free Speech Coalition, Inc. v. Paxton

#### Extractive Query (Query ID: 350)
> "FREE SPEECH COALITION, INC , et al PAXTON, certiorari to the united states court of appeals for Texas, like many States, prohibits distributing sexually explicit content to children In 2023, Texas ena..."
> 
> *Source: Free Speech Coalition, Inc. v. Paxton*

#### Generative Query (Query ID: 1003)
> "Can states require age verification for websites hosting sexually explicit content without violating the First Amendment?..."

#### Model Results for Extractive Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 5 | Free Speech Coalition, Inc. v. Paxton | ✓✓ Source |
| 2 | 4 | Mahmoud v. Taylor | ✗ Unrelated |
| 3 | 62 | Moody v. NetChoice, LLC | ✗ Unrelated |
| 4 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 5 | 77 | Gonzalez v. Trevino | ✗ Unrelated |
| 6 | 127 | United States v. Texas | ✓ Cited |
| 7 | 193 | Ysleta del Sur Pueblo v. Texas | ✗ Unrelated |
| 8 | 19 | United States v. Skrmetti | ✗ Unrelated |
| 9 | 117 | 303 Creative LLC v. Elenis | ✗ Unrelated |
| 10 | 257 | Greer v. United States | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 4 | 15 | Fuld v. Palestine Liberation Organization | ✗ Unrelated |
| 5 | 243 | Lombardo v. St. Louis | ✗ Unrelated |
| 6 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 7 | 334 | Atlantic Richfield Co. v. Christian | ✗ Unrelated |
| 8 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 9 | 42 | FDA v. Wages and White Lion Investments, LLC | ✗ Unrelated |
| 10 | 124 | Moore v. Harper | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 2 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 3 | 40 | Department of Education v. California | ✗ Unrelated |
| 4 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 5 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 6 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 7 | 295 | Tanzin v. Tanvir | ✗ Unrelated |
| 8 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 9 | 267 | Edwards v. Vannoy | ✗ Unrelated |
| 10 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |

#### Model Results for Generative Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 5 | Free Speech Coalition, Inc. v. Paxton | ✓✓ Source |
| 2 | 62 | Moody v. NetChoice, LLC | ✗ Unrelated |
| 3 | 117 | 303 Creative LLC v. Elenis | ✗ Unrelated |
| 4 | 19 | United States v. Skrmetti | ✗ Unrelated |
| 5 | 4 | Mahmoud v. Taylor | ✗ Unrelated |
| 6 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 7 | 308 | Agency for Int’l Development v. Alliance for Open Society | ✗ Unrelated |
| 8 | 184 | Vega v. Tekoh | ✗ Unrelated |
| 9 | 181 | Dobbs v. Jackson Women’s Health Organization | ✗ Unrelated |
| 10 | 314 | Bostock v. Clayton County | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 308 | Agency for Int’l Development v. Alliance for Open Society | ✗ Unrelated |
| 2 | 11 | Esteras v. United States | ✗ Unrelated |
| 3 | 330 | Romag Fasteners, Inc. v. Fossil Group, Inc. | ✗ Unrelated |
| 4 | 161 | Turkiye Halk Bankasi A.S. v. United States | ✗ Unrelated |
| 5 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 6 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 7 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 8 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 9 | 28 | Catholic Charities Bureau, Inc. v. Wisconsin Labor and Industry Review Comm’n. | ✗ Unrelated |
| 10 | 163 | New York v. New Jersey | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 4 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 5 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 6 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 7 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 8 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 9 | 295 | Tanzin v. Tanvir | ✗ Unrelated |
| 10 | 230 | Biden v. Missouri | ✗ Unrelated |

**Analysis:**
- **sentence-transformer**: ✅ Found source case at rank 1 for BOTH query types
- **legal-bert**: Source found at rank 70 (extractive), rank 72 (generative)
- **harvard-bert**: Source found at rank 14 (extractive), rank 23 (generative)

---

### Example: Loper Bright Enterprises v. Raimondo

#### Extractive Query (Query ID: 411)
> "LOPER BRIGHT ENTERPRISES et al RAIMONDO, SECRETARY OF COMMERCE, et al certiorari to the united states court of appeals for the district of columbia circuit The Court granted certiorari in these cases ..."
> 
> *Source: Loper Bright Enterprises v. Raimondo*

#### Generative Query (Query ID: 1069)
> "Here's a natural question that someone might search for based on the case summary:

"Should Chevron doctrine be overruled or clarified?"..."

#### Model Results for Extractive Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 66 | Loper Bright Enterprises v. Raimondo | ✓✓ Source |
| 2 | 31 | Seven County Infrastructure Coalition v. Eagle County | ✗ Unrelated |
| 3 | 164 | Axon Enterprise, Inc. v. FTC | ✗ Unrelated |
| 4 | 17 | EPA v. Calumet Shreveport Refining, L.L.C. | ✗ Unrelated |
| 5 | 78 | Moore v. United States | ✗ Unrelated |
| 6 | 312 | Liu v. SEC | ✗ Unrelated |
| 7 | 318 | Thole v. U. S. Bank N. A. | ✗ Unrelated |
| 8 | 255 | Nestlé USA, Inc. v. Doe | ✗ Unrelated |
| 9 | 299 | Trump v. Mazars USA, LLP | ✗ Unrelated |
| 10 | 273 | Carr v. Saul | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✓ Cited |
| 4 | 15 | Fuld v. Palestine Liberation Organization | ✗ Unrelated |
| 5 | 174 | West Virginia v. EPA | ✗ Unrelated |
| 6 | 243 | Lombardo v. St. Louis | ✗ Unrelated |
| 7 | 42 | FDA v. Wages and White Lion Investments, LLC | ✗ Unrelated |
| 8 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 9 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 10 | 334 | Atlantic Richfield Co. v. Christian | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 66 | Loper Bright Enterprises v. Raimondo | ✓✓ Source |
| 4 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 5 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✓ Cited |
| 6 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 7 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 8 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 9 | 297 | Sharp v. Murphy | ✗ Unrelated |
| 10 | 295 | Tanzin v. Tanvir | ✗ Unrelated |

#### Model Results for Generative Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 66 | Loper Bright Enterprises v. Raimondo | ✓✓ Source |
| 2 | 337 | CITGO Asphalt Refining Co. v. Frescati Shipping Co. | ✗ Unrelated |
| 3 | 31 | Seven County Infrastructure Coalition v. Eagle County | ✗ Unrelated |
| 4 | 106 | Macquarie Infrastructure Corp. v. Moab Partners, L. P. | ✗ Unrelated |
| 5 | 251 | Goldman Sachs Group, Inc. v. Arkansas Teacher Retirement System | ✗ Unrelated |
| 6 | 35 | Feliciano v. Department Of Transportation | ✗ Unrelated |
| 7 | 13 | Diamond Alternative Energy, LLC v. EPA | ✗ Unrelated |
| 8 | 17 | EPA v. Calumet Shreveport Refining, L.L.C. | ✗ Unrelated |
| 9 | 245 | HollyFrontier Cheyenne Refining, LLC v. Renewable Fuels Assn. | ✗ Unrelated |
| 10 | 78 | Moore v. United States | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 15 | Fuld v. Palestine Liberation Organization | ✗ Unrelated |
| 2 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 3 | 124 | Moore v. Harper | ✗ Unrelated |
| 4 | 243 | Lombardo v. St. Louis | ✗ Unrelated |
| 5 | 217 | Badgerow v. Walters | ✗ Unrelated |
| 6 | 334 | Atlantic Richfield Co. v. Christian | ✗ Unrelated |
| 7 | 308 | Agency for Int’l Development v. Alliance for Open Society | ✗ Unrelated |
| 8 | 163 | New York v. New Jersey | ✗ Unrelated |
| 9 | 98 | Consumer Financial Protection Bureau v. Community Financial Services Assn. of America, Ltd. | ✗ Unrelated |
| 10 | 77 | Gonzalez v. Trevino | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 2 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 3 | 66 | Loper Bright Enterprises v. Raimondo | ✓✓ Source |
| 4 | 40 | Department of Education v. California | ✗ Unrelated |
| 5 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✓ Cited |
| 6 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 7 | 185 | New York State Rifle & Pistol Assn., Inc. v. Bruen | ✗ Unrelated |
| 8 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 9 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 10 | 295 | Tanzin v. Tanvir | ✗ Unrelated |

**Analysis:**
- **sentence-transformer**: ✅ Found source case at rank 1 for BOTH query types
- **legal-bert**: Source found at rank 64 (generative)
- **harvard-bert**: Source found at rank 3 (extractive), rank 3 (generative)

---

### Example: Trump v. Anderson

#### Extractive Query (Query ID: 457)
> "ANDERSON et al certiorari to the supreme court of colorado Six Colorado voters (respondents here) fled a petition in Colorado state court against former President Donald J Trump and Colorado Secretary..."
> 
> *Source: Trump v. Anderson*

#### Generative Query (Query ID: 1115)
> "Here's a natural question that someone might search for based on this case summary:

"Can a former US President be barred from running for President again under Section 3 of the Fourteenth Amendment i..."

#### Model Results for Extractive Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 112 | Trump v. Anderson | ✓✓ Source |
| 2 | 300 | Trump v. Vance | ✓ Cited |
| 3 | 294 | Carney v. Adams | ✗ Unrelated |
| 4 | 305 | Chiafalo v. Washington | ✓ Cited |
| 5 | 1 | Trump v. CASA, Inc. | ✗ Unrelated |
| 6 | 65 | Fischer v. United States | ✗ Unrelated |
| 7 | 63 | Trump v. United States | ✓ Cited |
| 8 | 123 | Counterman v. Colorado | ✗ Unrelated |
| 9 | 313 | Department of Homeland Security v. Regents of Univ. of Cal. | ✗ Unrelated |
| 10 | 237 | Brnovich v. Democratic National Committee | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 4 | 15 | Fuld v. Palestine Liberation Organization | ✗ Unrelated |
| 5 | 243 | Lombardo v. St. Louis | ✗ Unrelated |
| 6 | 334 | Atlantic Richfield Co. v. Christian | ✗ Unrelated |
| 7 | 124 | Moore v. Harper | ✗ Unrelated |
| 8 | 42 | FDA v. Wages and White Lion Investments, LLC | ✗ Unrelated |
| 9 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |
| 10 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 4 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 5 | 298 | McGirt v. Oklahoma | ✗ Unrelated |
| 6 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 7 | 295 | Tanzin v. Tanvir | ✗ Unrelated |
| 8 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 9 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 10 | 316 | Lomax v. Ortiz-Marquez | ✗ Unrelated |

#### Model Results for Generative Query

**sentence-transformer**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 112 | Trump v. Anderson | ✓✓ Source |
| 2 | 300 | Trump v. Vance | ✓ Cited |
| 3 | 305 | Chiafalo v. Washington | ✓ Cited |
| 4 | 63 | Trump v. United States | ✓ Cited |
| 5 | 1 | Trump v. CASA, Inc. | ✗ Unrelated |
| 6 | 261 | Van Buren v. United States | ✗ Unrelated |
| 7 | 313 | Department of Homeland Security v. Regents of Univ. of Cal. | ✗ Unrelated |
| 8 | 294 | Carney v. Adams | ✗ Unrelated |
| 9 | 65 | Fischer v. United States | ✗ Unrelated |
| 10 | 289 | Trump v. New York | ✗ Unrelated |

**legal-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 2 | 40 | Department of Education v. California | ✗ Unrelated |
| 3 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 4 | 15 | Fuld v. Palestine Liberation Organization | ✗ Unrelated |
| 5 | 334 | Atlantic Richfield Co. v. Christian | ✗ Unrelated |
| 6 | 163 | New York v. New Jersey | ✗ Unrelated |
| 7 | 42 | FDA v. Wages and White Lion Investments, LLC | ✗ Unrelated |
| 8 | 223 | United States v. Tsarnaev | ✗ Unrelated |
| 9 | 124 | Moore v. Harper | ✗ Unrelated |
| 10 | 185 | New York State Rifle & Pistol Assn., Inc. v. Bruen | ✗ Unrelated |

**harvard-bert**:
| Rank | Case ID | Case Name | Relevance |
|------|---------|-----------|-----------|
| 1 | 275 | Tandon v. Newsom | ✗ Unrelated |
| 2 | 172 | Arellano v. McDonough | ✗ Unrelated |
| 3 | 66 | Loper Bright Enterprises v. Raimondo | ✗ Unrelated |
| 4 | 332 | Ramos v. Louisiana Revisions : 8/28/25 | ✗ Unrelated |
| 5 | 253 | United States v. Arthrex, Inc. | ✗ Unrelated |
| 6 | 40 | Department of Education v. California | ✗ Unrelated |
| 7 | 185 | New York State Rifle & Pistol Assn., Inc. v. Bruen | ✗ Unrelated |
| 8 | 296 | Roman Catholic Diocese of Brooklyn v. Cuomo | ✗ Unrelated |
| 9 | 267 | Edwards v. Vannoy | ✗ Unrelated |
| 10 | 326 | New York State Rifle & Pistol Assn., Inc. v. City of New York | ✗ Unrelated |

**Analysis:**
- **sentence-transformer**: ✅ Found source case at rank 1 for BOTH query types
- **legal-bert**: ❌ Failed to find source case in top-100 for both queries
- **harvard-bert**: ❌ Failed to find source case in top-100 for both queries

---

