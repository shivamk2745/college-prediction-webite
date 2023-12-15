import pandas as pd
import numpy as np
import csv
import math
from sklearn import neighbors, datasets
from numpy.random import permutation
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from pandas import DataFrame

data = pd.read_csv('cutoffobc.csv',encoding= 'unicode_escape')
processed_data = data[['kcet','college']]
random_indices = permutation(data.index)
test_cutoff = math.floor(len(data)/5)
test = processed_data.loc[random_indices[1:test_cutoff]]
train = processed_data.loc[random_indices[test_cutoff:]]
train_output_data = train['college']
train_input_data = train
train_input_data = train_input_data.drop('college', axis=1)
test_output_data = test['college']
test_input_data = test
test_input_data = test_input_data.drop('college', axis=1)


college_links = {
    "College of Engineering, Pune": "https://www.coep.org.in/",
    
    'Pune Institute of Computer Technology, Dhankavdi, Pune': 'https://pict.edu/',
    'Walchand College of Engineering, Sangli': 'http://www.walchandsangli.ac.in/',
    "Bansilal Ramnath Agarawal Charitable Trust's Vishwakarma Institute of Technology, Bibwewadi, Pune": 'https://www.vit.edu/',
    "Bharati Vidyapeeth's College of Engineering,Lavale, Pune": 'http://bvcoe.bharatividyapeeth.edu/',
    'International Institute of Information Technology , Pune.': 'https://www.isquareit.edu.in/',
    'Shri Ramdeobaba College of Engineering and Management, Nagpur': 'http://www.rknec.edu/',
    "Shri Vile Parle Kelvani Mandal's Dwarkadas J. Sanghvi College of Engineering, Vile Parle, Mumbai": 'https://www.djsce.ac.in/',
    'Pimpri Chinchwad Education Trust, Pimpri Chinchwad College of Engineering, Pune': 'http://www.pccoepune.com/',
    'Government College of Engineering, Amravati': 'https://www.gcoea.ac.in/',
    'MKSSS\'s Cummins College of Engineering for Women, Karvenagar, Pune': 'https://www.cumminscollege.org/',
    "B.R.A.C.T's Vishwakarma Institute of Information Technology, Kondhwa (Bk.), Pune": 'https://www.viit.ac.in/',
    'Bharati Vidyapeeth College of Engineering, Navi Mumbai': 'https://bvcoenm.edu.in/',
    'K J Somaiya Institute of Engineering and Information Technology, Sion, Mumbai': 'https://kjsit.somaiya.edu.in/en',
    'MIT Academy of Engineering, Alandi, Pune': 'https://mitaoe.ac.in/',
    'Laxminarayan Institute of Technology, Nagpur': 'https://litnagpur.in/',
    'Vivekanand Education Society\'s Institute of Technology, Chembur, Mumbai': 'https://vesit.ves.ac.in/',
    'Government College of Engineering & Research, Avasari Khurd': 'https://www.gcoeara.ac.in/',
    'Pimpri Chinchwad Education Trust\'s Pimpri Chinchwad College Of Engineering And Research, Ravet': 'https://www.pccoer.com/',
    'Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded': 'https://www.sggs.ac.in/',
    'Thadomal Shahani Engineering College, Bandra, Mumbai': 'https://tsec.edu/',
    'Vidyalankar Institute of Technology,Wadala, Mumbai': 'https://vit.edu.in/',
    'Dr. D. Y. Patil Pratishthan\'s D.Y.Patil College of Engineering Akurdi, Pune': 'https://www.dypcoeakurdi.ac.in/',
    'Agnel Charities\' FR. C. Rodrigues Institute of Technology, Vashi, Navi Mumbai': 'https://fcrit.ac.in/',
    'All India Shri Shivaji Memorial Society\'s College of Engineering, Pune': 'https://aissmscoe.com/',
    'Sinhgad College of Engineering, Vadgaon (BK), Pune': 'http://cms.sinhgad.edu/sinhgad_engineering_institutes/vadgaon_scoe/about.aspx',
    'Fr. Conceicao Rodrigues College of Engineering, Bandra,Mumbai': 'http://www.frcrce.ac.in/',
    'Progressive Education Society\'s Modern College of Engineering, Pune': 'https://moderncoe.edu.in/',
    'Government College of Engineering, Chandrapur': 'http://www.gcoec.ac.in/gcoec/',
    'Ramrao Adik Edu Soc, Ramarao Adik Institute of Tech., Navi Mumbai': 'https://dypatil.edu/engineering/index.php',
    'Government College of Engineering, Jalgaon': 'https://www.gcoej.ac.in/',
    "Jaywant Shikshan Prasarak Mandal's,Rajarshi Shahu College of Engineering, Tathawade, Pune": 'https://www.jspmrscoe.edu.in/',
    "All India Shri Shivaji Memorial Society's Institute of Information Technology,Pune": 'https://aissmsioit.org/',
    'Shri Sant Gajanan Maharaj College of Engineering,Shegaon': 'https://www.ssgmce.ac.in/',
    'Dr. D.Y.Patil Institute of Engineering, Management & Reseach, Akurdi, Pune': 'https://www.dypiemr.ac.in/',
    'Manjara Charitable Trust\'s Rajiv Gandhi Institute of Technology, Mumbai': 'https://www.rgit.ac.in/',
    'K. K. Wagh Institute of Engineering Education and Research, Nashik': 'https://engg.kkwagh.edu.in/',
    'Modern Education Society\'s College of Engineering, Pune': 'https://mescoe.mespune.org/',
    'University Department of Chemical Technology, Aurangabad': 'http://www.bamu.ac.in/dept-of-chemical-technology/Home.aspx',
    'Rajiv Gandhi College of Engineering, At Post Karjule Hariya Tal.Parner, Dist.Ahmednagar': 'https://www.rgcoe.org/',
    'Dr. D. Y. Patil Vidya Pratishthan Society Dr .D. Y. Patil Institute of Engineering & Technology, Pimpri, Pune': 'https://engg.dypvp.edu.in/',
    'Government Engineering College, Yavatmal': 'https://gcoey.ac.in/',
    "K. E. Society's Rajarambapu Institute of Technology, Walwa, Sangli": 'https://www.ritindia.edu/',
    'Bharati Vidyapeeth\'s College of Engineering for Women, Katraj, Dhankawadi, Pune': 'http://coewpune.bharatividyapeeth.edu/',
    'Yeshwantrao Chavan College of Engineering,Wanadongri, Nagpur': 'https://www.ycce.edu/',
    'Sinhgad Technical Education Society\'s Smt. Kashibai Navale College of Engineering, Vadgaon, Pune': 'http://cms.sinhgad.edu/sinhgad_engineering_institutes/skncoe_vadgaon/institute_details.aspx',
    'Xavier Institute Of Engineering C/O Xavier Technical Institute, Mahim, Mumbai': 'https://www.xavier.ac.in/',
    'Marathwada Mitra Mandal\'s College of Engineering, Karvenagar, Pune': 'https://www.mmcoe.edu.in/',
    'Walchand Institute of Technology, Solapur': 'https://witsolapur.org/',
    "JSPM'S Jaywantrao Sawant College of Engineering,Pune": 'https://jspmjscoe.edu.in/',
    'Kolhapur Institute of Technology\'s College of Engineering(Autonomous), Kolhapur': 'https://www.kitcoek.in/',
    'G.H.Raisoni Academy of Engineering & Technology, Nagpur': 'https://ghrietn.raisoni.net/',
    'St. Francis Institute of Technology, Borivali, Mumbai': 'https://www.sfit.ac.in/',
    'D. Y. Patil College of Engineering, Ambi, Talegaon, Maval': 'https://dypatiluniversitypune.edu.in/school-of-engineering-ambi.php',
    'JSPM\'s Imperial College of Engineering and Research, Wagholi, Pune': 'https://jspmicoer.edu.in/',
    'JSPM Narhe Technical Campus, Pune.': 'https://jspmntc.edu.in/',
    'Sanjivani Rural Education Society\'s Sanjivani College of Engineering, Kopargaon': 'https://sanjivanicoe.org.in/',
    'Nashik District Maratha Vidya Prasarak Samaj\'s Karmaveer Adv.Babaurao Ganpatrao Thakare College of Engineering, Nashik': 'https://kbtcoe.org/',
    'Vidya Pratishthan\'s Kamalnayan Bajaj Institute of Engineering & Technology, Baramati Dist.Pune': 'https://www.vpkbiet.org/',
    'TSSMS\'s Pd. Vasantdada Patil Institute of Technology, Bavdhan, Pune': 'https://pvpittssm.edu.in/',
    'Guru Gobind Singh College of Engineering & Research Centre, Nashik.': 'https://www.ggsf.edu.in/placement.php',
    'Dattajirao Kadam Technical Education Society\'s Textile & Engineering Institute, Ichalkaranji.': 'https://www.dkte.ac.in/',
    'Nutan Maharashtra Vidya Prasarak Mandal, Nutan Maharashtra Institute of Engineering &Technology, Talegaon station, Pune': 'https://www.nmiet.edu.in/',
    'Sinhgad Technical Education Society, Sinhgad Institute of Technology and Science, Narhe (Ambegaon)': 'http://cms.sinhgad.edu/sinhgad_engineering_institutes/sits_narhetechnicalcampus/sits_nt_aboutus.aspx',
    'Ankush Shikshan Sanstha\'s G.H.Raisoni College of Engineering, Nagpur': 'https://ghrce.raisoni.net/',
    "Rizvi Education Society's Rizvi College of Engineering, Bandra,Mumbai": 'https://eng.rizvi.edu.in/',
    'Dhole Patil Education Society, Dhole Patil College of Engineering, Wagholi, Tal. Haveli': 'https://dpcoepune.edu.in/',
    'Sinhagad Institute of Technology, Lonavala': 'http://cms.sinhgad.edu/sinhgad_engineering_institutes/sit_lonavala/about_institute.aspx',
    'Pune District Education Association\'s College of Engineering, Pune': 'http://www.pdeacoem.org/',
    'Indira College of Engineering & Management, Pune': 'https://indiraicem.ac.in/',
    'Krushi Jivan Vikas Pratishthan, Ballarpur Institute of Technology, Mouza Bamni': 'https://www.indiastudychannel.com/colleges/36738-krushi-jivan-vikas-pratishthan-ballarpur-institute-technology-mouza-bamni-chandrapur',
    'Amrutvahini Sheti & Shikshan Vikas Sanstha\'s Amrutvahini College of Engineering, Sangamner': 'http://www.avcoe.org/',
    'Prof. Ram Meghe Institute of Technology & Research, Amravati': 'https://mitra.ac.in/',
    'Zeal Education Society\'s Zeal College of Engineering & Reserch, Narhe, Pune': 'https://zcoer.in/',
    'Shivnagar Vidya Prasarak Mandal\'s College of Engineering, Malegaon-Baramati': 'https://www.engg.svpm.org.in/',
    'Department of Technology, Shivaji University, Kolhapur': 'https://apps.unishivaji.ac.in/',
    'TSSM\'s Bhivarabai Sawant College of Engineering and Research, Narhe, Pune': 'https://tssm.edu.in/',
    'M.G.M.\'s Jawaharlal Nehru Engineering College, Aurangabad': 'https://www.jnec.org/',
    'G.H.Raisoni College of Engineering & Management, Wagholi, Pune': 'https://ghrcem.raisoni.net/',
    'S.I.E.S. Graduate School of Technology, Nerul, Navi Mumbai': 'https://siesgst.edu.in/',
    'Dr. D. Y. Patil School OF Engineering, Lohegaon, Pune': 'https://adypsoe.in/',
    'Marathwada Mitra Mandal\'s Institute of Technology, Lohgaon, Pune': 'https://mmit.edu.in/',
    'G. H.Raisoni Institute of Engineering and Technology, Wagholi, Pune': 'https://ghrcem.raisoni.net/',
    'Nutan College of Engineering and Research, Talegaon Dabhade Tal. Maval, Pune': 'https://www.ncerpune.in/',
    'D. Y. Patil Institute of Engineering and Technology, Ambi': 'https://dypatiluniversitypune.edu.in/school-of-engineering-ambi.php',
    'NBN Sinhgad Technical Institutes Campus, Pune': 'http://nbnstic.sinhgad.edu/',
    'Sandip Foundation\'s, Sandip Institute of Engineering & Management, Nashik': 'https://siem.sandipfoundation.org/',
    'Sandip Foundation, Sandip Institute of Technology and Research Centre, Mahiravani, Nashik': 'https://sitrc.sandipfoundation.org/',
    'Sinhgad Academy of Engineering, Kondhwa (BK) Kondhwa-Saswad Road, Pune': 'http://cms.sinhgad.edu/sinhgad_engineering_institutes/saoe/about_us.aspx',
    'Annasaheb Dange College of Engineering and Technology, Ashta, Sangli': 'https://www.adcet.ac.in/',
    'Smt. Indira Gandhi College of Engineering, Navi Mumbai': 'http://sigce.edu.in/',
    "Jawahar Education Society's Annasaheb Chudaman Patil College of Engineering,Kharghar, Navi Mumbai": 'https://www.acpce.org/',
    'ST. Vincent Pallotti College of Engineering & Technology, Nagpur': 'https://www.stvincentngp.edu.in/',
    'Bharati Vidyapeeth\'s College of Engineering, Kolhapur': 'http://coekolhapur.bharatividyapeeth.edu/',
    'M.S. Bidve Engineering College, Latur': 'https://www.msbecl.ac.in/',
    'Atharva College of Engineering,Malad(West),Mumbai': 'https://atharvacoe.ac.in/',
    'Guru Nanak Institute of Engineering & Technology,Kalmeshwar, Nagpur': 'http://gnietedu.com/',
    'Pravara Rural College of Engineering, Loni, Pravaranagar, Ahmednagar.': 'https://www.pravaraengg.org.in/',
    'Genba Sopanrao Moze Trust Parvatibai Genba Moze College of Engineering,Wagholi, Pune': 'https://www.pgmozecoepune.in/',
    'Gramin College of Engineering, Vishnupuri, Nanded': 'https://www.pgmozecoepune.in/',
    'N.Y.S.S.\'s Datta Meghe College of Engineering, Airoli, Navi Mumbai': 'https://www.dmce.ac.in/',
    'Dr. D. Y. Patil Educational Academy\'s, D.Y.Patil School of Engineering Academy, Ambi': 'https://bcud.unipune.ac.in/utilities/college_search/CEGP019340_ENG/Pune_University_College',
    'M.G.M.\'s College of Engineering and Technology, Kamothe, Navi Mumbai': 'http://www.mgmmumbai.ac.in/mgmcet/home',
    'Pune Vidyarthi Griha\'s College Of Engineering, Nashik': 'https://www.pvgcoenashik.org/',
    'Dr. D Y Patil Pratishthan\'s College of Engineering, Kolhapur': 'https://collegedunia.com/college/60250-dr-dy-patil-pratishthans-college-of-engineering-kolhapur',
    'K.D.M. Education Society, Vidharbha Institute of Technology,Umred Road ,Nagpur': 'https://vitnagpur.in/',
    'G. S. Mandal\'s Maharashtra Institute of Technology, Aurangabad': 'https://engg.mit.asia/',
    'Matoshri College of Engineering and Research Centre, Eklahare, Nashik': 'https://engg.matoshri.edu.in/',
    'MET Bhujbal Knowledge City MET League\'s Engineering College, Adgaon, Nashik.': 'https://metbhujbalknowledgecity.ac.in/metengg/',
    'Dr. Babasaheb Ambedkar College of Engineering and Research, Wanadongri, Nagpur': 'https://www.shiksha.com/college/dr-babasaheb-ambedkar-college-of-engineering-and-research-nagpur-47571',
    'Vidyavardhini\'s College of Engineering and Technology, Vasai': 'https://vcet.edu.in/',
    'Jayawant Shikshan Prasarak Mandal, Bhivarabai Sawant Institute of Technology & Research, Wagholi': 'https://jspmbsiotr.edu.in/',
    'Deogiri Institute of Engineering and Management Studies, Aurangabad': 'https://www.dietms.org/',
    'Cummins College of Engineering For Women, Sukhali (Gupchup), Tal. Hingna Hingna Nagpur': 'https://cumminscollege.edu.in/',
    'Pravara Rural Education Society\'s Sir Visvesvaraya Institute of Technology, Chincholi Dist. Nashik': 'https://svitnashik.in/',
    'A. P. Shah Institute of Technology, Thane': 'https://www.apsit.edu.in/',
    'K. C. E. Societys College of Engineering and Information Technology, Jalgaon': 'https://coem.ac.in/',
    'Rasiklal M. Dhariwal Sinhgad Technical Institutes Campus, Warje, Pune.': 'http://rmdstic.sinhgad.edu/',
    'P.K. Technical Campus, Pune.': 'https://pkinstitute.edu.in/',
    'Padmabhushan Vasantdada Patil Pratishthans College of Engineering, Sion, Mumbai': 'https://pvppcoe.ac.in/',
    'CSMSS Chh. Shahu College of Engineering, Aurangabad': 'https://www.csmssengg.org/',
    'Metropolitan Institute of Technology & Management, Sukhalwad, Sindhudurg.': 'https://mitm.ac.in/',
    'Rayat Shikshan Sanstha\'s Karmaveer Bhaurao Patil College of Engineering, Satara': 'https://kbpcoes.edu.in/',
    'Terna Engineering College, Nerul, Navi Mumbai': 'https://ternaengg.ac.in/',
    'Gokhale Education Society\'s, R.H. Sapat College of Engineering, Management Studies and Research, Nashik': 'https://ges-coengg.org/',
    'Mahatma Education Society\'s Pillai College of Engineering, New Panvel': 'https://www.pce.ac.in/',
    'Everest Education Society, Group of Institutions (Integrated Campus), Ohar': 'https://www.eescoet.org/',
    'MAEER\'s MIT College of Railway Engineering and Research, Jamgaon, Barshi': 'https://mitcorer.edu.in/',
    'KJEI\'s Trinity Academy of Engineering, Yewalewadi, Pune': 'https://www.kjei.edu.in/tae/',
    'Genba Sopanrao Moze College of Engineering, Baner-Balewadi, Pune': 'https://www.gsmozecoe.org/',
    'D.Y. Patil College of Engineering and Technology, Kolhapur': 'https://dypgroup.edu.in/',
    'Dattakala Group Of Institutions, Swami - Chincholi Tal. Daund Dist. Pune': 'https://www.dattakala.edu.in/',
    'SKN Sinhgad Institute of Technology & Science, Kusgaon(BK),Pune..': 'http://cms.sinhgad.edu/sinhgad_engineering_institutes/sknsits_lonavala/about_us.aspx',
    'G.M.Vedak Institute of Technology, Tala, Raigad.': 'https://gmvit.com/',
    'Jagadambha Bahuuddeshiya Gramin Vikas Sanstha\'s Jagdambha College of Engineering and Technology, Yavatmal': 'https://www.jcoet.ac.in/profile/1',
    'V M Institute of Engineering and Technology, Dongargaon, Nagpur': 'http://vmitgpg.com/',
    'Siddhivinayak Technical Campus, School of Engineering & Research Technology, Shirasgon, Nile': 'http://stc.org.in/',
    'Sant Gajanan Maharaj College of Engineering, Gadhinglaj': 'http://sgmcoe.in/',
    'Dr.D.Y.Patil College Of Engineering & Innovation,Talegaon': 'https://www.dypcoei.edu.in/',
    'Universal College of Engineering & Research, Sasewadi': 'https://www.ucoer.edu.in/',
    'Sipna Shikshan Prasarak Mandal College of Engineering & Technology, Amravati': 'https://sipnaengg.ac.in/',
    'Late Shri. Vishnu Waman Thakur Charitable Trust, Viva Institute of Technology, Shirgaon': 'https://www.viva-technology.org/New/',
    'Vishwaniketan\'s Institute of Management Entrepreneurship and Engineering Technology(i MEET), Khalapur Dist Raigad': 'https://vimeet.vishwaniketan.edu.in/',
    'Navsahyadri Education Society\'s Group of Institutions': 'https://navsahyadri.edu.in/',
    "D.Y.Patil Education Society's,D.Y.Patil Technical Campus, Faculty of Engineering & Faculty of Management,Talsande,Kolhapur.": 'https://foet.dypgroup.edu.in/',
    'K.V.N. Naik S. P. Sansth\'s Loknete Gopinathji Munde Institute of Engineering Education & Research, Nashik.': 'https://collegedunia.com/college/61024-kvn-naik-sp-sansthas-polytechnic-nashik',
    'College of Engineering and Technology ,North Maharashtra Knowledge City, Jalgaon': 'https://sscoetjalgaon.ac.in/',
    'K. J.\'s Educational Institut Trinity College of Engineering and Research, Pisoli, Haveli': 'https://kjei.edu.in/tcoer/',
    'Jawahar Education Society\'s Institute of Technology, Management & Research, Nashik.': 'https://www.jitnashik.edu.in/',
    'I.S.B.& M. School of Technology, Nande Village': 'https://bcud.unipune.ac.in/utilities/college_search/CEGP015950_ENG/Pune_University_College',
    "Alard Charitable Trust's Alard College of Engineering and Management, Pune": 'https://www.alardcollegeofengineering.com/',
    'Lokmanya Tilak College of Engineering, Kopar Khairane, Navi Mumbai': 'https://www.ltce.in/',
    'G. H. Raisoni College of Engineering and Management, Ahmednagar': 'https://bcud.unipune.ac.in/utilities/college_search/CEGA016960_ENG/Pune_University_College',
    'Prof Ram Meghe College of Engineering and Management, Badnera': 'https://prmceam.ac.in/',
    "Shri Shivaji Education Society's College of Engineering and Technology, Akola": 'https://coeta.ac.in/',
    "Peoples Education Society's College of Engineering, Aurangabad": 'http://pescoe.ac.in/',
    'Samarth Group of Institutions, Bangarwadi, Post Belhe Tal. Junnar Dist. Pune': 'https://engg.sreir.org/',
    'Lokmanya Tilak Jankalyan Shikshan Sastha, Priyadarshini J. L. College Of Engineering, Nagpur': 'https://pjlce.edu.in/',
    'Maitraya Education Society, Nagarjuna Institute of Engineering Technology & Management, Nagpur': 'https://nietm.in/',
    "Vidya Prasarini Sabha's College of Engineering & Technology, Lonavala": 'http://www.vps-cet.com/',
    'G. H. Raisoni Institute of Business Management,Jalgaon': 'https://ghribmjal.raisoni.net/',
    "Hon. Shri. Babanrao Pachpute Vichardhara Trust, Group of Institutions (Integrated Campus)-Parikrama, Kashti Shrigondha": 'https://www.collegedekho.com/colleges/honshri-babanrao-pachpute-vichardhara-trusts-group-of-institutions',
    'Fabtech Technical Campus College of Engineering and Research, Sangola': 'https://ftccoe.ac.in/',
    'Keystone School of Engineering, Pune': 'https://www.keystoneschoolofengineering.com/',
    'K.D.K. College of Engineering, Nagpur': 'https://www.kdkce.edu.in/',
    "Ankush Shikshan Sanstha's G. H. Raisoni Institute of Engineering & Technology, Nagpur": 'https://ghrietn.raisoni.net/',
    "Aldel Education Trust's St. John College of Engineering & Management, Vevoor, Palghar": 'https://www.aldel.in/',
    'Bajaj Institute of Technology, Wardha': 'https://bitwardha.ac.in/',
    'Anantrao Pawar College of Engineering & Research, Pune': 'https://abmspcoerpune.org/Homepage.aspx',
    "Sanmarg Shikshan Sanstha's Smt. Radhikatai Pandav College of Engineering, Nagpur": 'https://srpce.ac.in/',
    "Vishvatmak Jangli Maharaj Ashram Trust's Vishvatmak Om Gurudev College of Engineering, Mohili-Aghai, Shahpur.": 'https://www.icbse.com/colleges/vishwatmak-jangli-maharaj-trust-vishwatmak-om-gurudev-colleg-56lz5e',
    "Kai Amdar Bramhadevdada Mane Shikshan & Samajik Prathistan's Bramhadevdada Mane Institute of Technology, Solapur": 'https://bmitsolapur.org/',
    'Mahatma Gandhi Missions College of Engineering, Hingoli Rd, Nanded.': 'http://www.mgmcen.ac.in/',
    'Vidya Niketan College of Engineering, Bota Sangamner': 'https://vidyaniketanglobal.com/',
    "Hindi Seva Mandal's Shri Sant Gadgebaba College of Engineering & Technology, Bhusawal": 'https://www.ssgbcoet.com/',
    'Lokmanya Tilak Jankalyan Shikshan Sanstha, Priyadarshani College of Engineering, Nagpur': 'https://www.pcenagpur.edu.in/',
    'Jaihind College Of Engineering,Kuran': 'https://jaihind.edu.in/jcoe/',
    "Shri Vithal Education and Research Institute's College of Engineering, Pandharpur": 'https://www.collegedekho.com/colleges/college-of-engineering-pandharpur',
    "Janata Shikshan Prasarak Mandal’s Babasaheb Naik College Of Engineering, Pusad": 'https://www.bncoepusad.ac.in/',
    "Godavari Foundation's Godavari College Of Engineering, Jalgaon": 'https://www.gfgcoe.in/',
    "Kalyani Charitable Trust, Late Gambhirrao Natuba Sapkal College of Engineering, Anjaneri, Trimbakeshwar Road, Nashik": 'https://www.lgnscoe.sapkalknowledgehub.org/',
    "Vidarbha Bahu-Uddeshiya Shikshan Sanstha's Tulshiramji Gaikwad Patil College of Engineering & Technology, Nagpur": 'http://www.tgpcet.com/',
    "Adsul's Technical Campus, Chas Dist. Ahmednagar": 'http://www.adsulstechnicalcampus.com/',
    'Yashoda Technical Campus, Wadhe, Satara.': 'https://www.yes.edu.in/',
    'Priyadarshini Bhagwati College of Engineering, Harpur Nagar, Umred Road,Nagpur': 'https://pbcoe.edu.in/',
    "Gondia Education Society's Manoharbhai Patel Institute Of Engineering & Technology, Shahapur, Bhandara": 'http://mietbhandara.ac.in/#',
    'Bapurao Deshmukh College of Engineering, Sevagram': 'https://collegedunia.com/college/13001-bapurao-deshmukh-college-of-engineering-bdce-wardha',
    "Dr. Ashok Gujar Technical Institute's Dr. Daulatrao Aher College of Engineering, Karad": 'http://dacoe.ac.in/',
    "P. R. Pote (Patil) Education & Welfare Trust's Group of Institution(Integrated Campus), Amravati": 'https://prpotepatilengg.ac.in/',
    'Hi-Tech Institute of Technology, Aurangabad': 'http://hitechengg.edu.in/',
    'Sahyadri Valley College of Engineering & Technology, Rajuri, Pune.': 'https://sahyadrivalleycollege.com/',
    "Shri Pandurang Pratishtan, Karmayogi Engineering College, Shelve, Pandharpur": 'http://www.karmayogi.org.in/about.html',
    "K.J.'s Educational Institute's K.J.College of Engineering & Management Research, Pisoli": 'https://kjei.edu.in/kjcoemr/',
    "Vidya Vikas Pratishthan Institute of Engineering and Technology, Solapur": 'https://collegedunia.com/college/28896-vidya-vikas-pratishthan-institute-of-engineering-and-technology-solapur',
    "Paramhansa Ramkrishna Maunibaba Shikshan Santha's , Anuradha Engineering College, Chikhali": 'https://www.aecc.ac.in/anuradha/admission/home.php',
    "Mahavir Education Trust's Shah & Anchor Kutchhi Engineering College, Mumbai": 'https://www.sakec.ac.in/',
    "Lokmanya Tilak Jankalyan Shikshan Sanstha's , Priyadarshini Institute of Engineering and Technology, Nagpur": 'https://www.shiksha.com/college/lokmanya-tilak-jankalyan-shikshan-sanstha-s-priyadarshini-college-of-engineering-nagpur-7445',
    "Shanti Education Society, A.G. Patil Institute of Technology, Soregaon, Solapur(North)": 'http://www.agpit.edu.in/',
    "Brahma Valley College of Engineering & Research, Trimbakeshwar, Nashik": 'http://www.engineering.brahmavalley.com/',
    "Loknete Hanumantrao Charitable Trust's Adarsh Institute of Technology and Research Centre, Vita,Sangli": 'http://www.aitrcvita.edu.in/',
    'WATUMULL INSTITUTE OF ELECTRONICS ENGINEERING & COMPUTER TECHNOLOGY, ULHASNAGAR': 'http://www.watumull.edu/new/',
    'Sanghavi College of Engineering, Varvandi, Nashik.': 'https://engineering.shreemahavir.org/',
    'Ideal Institute of Technology, Wada, Dist.Thane': 'http://www.idealwada.com/about_ideal.php',
    "SNJB's Late Sau. Kantabai Bhavarlalji Jain College of Engineering, (Jain Gurukul), Neminagar,Chandwad,(Nashik)": 'https://snjb.org/engineering/',
    'Konkan Gyanpeeth College of Engineering, Karjat': 'https://kgce.edu.in/',
    "Shriram Gram Vikas Shikshan Sanstha, Vilasrao Deshmukh College of Engineering and Technology, Nagpur": 'https://collegedunia.com/college/19191-vilasrao-deshmukh-college-of-engineering-and-technology-vdcet-nagpur',
    "Mahatma Basaweshwar Education Society's College of Engineering, Ambejogai": 'http://www.coea.ac.in/',
    "Saraswati Education Society's Saraswati College of Engineering,Kharghar Navi Mumbai": 'https://engineering.saraswatikharghar.edu.in/',
    'Mauli Group of Institutions, College of Engineering and Technology, Shegaon.': 'http://mcoet.mauligroup.org/',
    'Shree Ramchandra College of Engineering, Lonikand,Pune': 'http://www.srespune.org/',
    "P. R. Pote Patil Institute of Engineering & Research, At Kathora, Amravati": 'https://prpotepatilengg.ac.in/',
    'Shivajirao S. Jondhale College of Engineering, Dombivali,Mumbai': 'https://shivajiraojondhalecoe.org.in/',
    "Sanmarg Shikshan Sanstha, Mandukarrao Pandav College of Engineering, Bhandara": 'https://www.shiksha.com/college/mandukarrao-pandav-college-of-engineering-bhandara-nagpur-63379',
    'Rajiv Gandhi College of Engineering & Research, Hingna Road, Nagpur': 'https://rgcer.edu.in/',
    "Jagadamba Education Soc. Nashik's S.N.D. College of Engineering & Reserch, Babulgaon": 'https://sndcoe.ac.in/',
    'Kavi Kulguru Institute of Technology & Science, Ramtek': 'https://kits.edu/',
    'Aditya Engineering College , Beed': 'https://www.aec.edu.in/',
    "Vidharbha Bahu uddeshiya Shikshan Sanstha's Abha Gaikwad – Patil College of Engineering, Nagpur": 'http://www.tgpcet.com/',
    "Shramsadhana Bombay Trust, College of Engineering & Technology, Jalgaon": 'https://sscoetjalgaon.ac.in/',
    'S K N Sinhgad College of Engineering, Korti Tal. Pandharpur Dist Solapur': 'http://www.sknscoe.ac.in/',
    "Dwarka Bahu Uddeshiya Gramin Vikas Foundation, Rajarshri Shahu College of Engineering, Buldhana": 'http://rsce.ac.in/',
    'International Centre of Excellence in Engineering & Management, Aurangabad.': 'https://iceemabad.com/',
    "Lokmanya Tilak Jankalyan Shiksan Sanstha, Priyadarshini Indira Gandhi College of Engineering, Nagpur": 'https://www.pcenagpur.edu.in/',
    "Padmashri Dr. V.B. Kolte College of Engineering, Malkapur, Buldhana": 'https://coemalkapur.ac.in/',
    "Vighnaharata Trust's Shivajirao S. Jondhale College of Engineering & Technology, Shahapur, Asangaon, Dist Thane": 'https://shivajiraojondhalecoe.org.in/',
    "STMEI's Sandipani Technical Campus-Faculty of Engineering, Latur.": 'https://collegedunia.com/college/28702-sandipani-technical-campus-faculty-of-engineering-stmei-latur/faculty',
    'Ahmednagar Jilha Maratha Vidya Prasarak Samajache, Shri. Chhatrapati Shivaji Maharaj College of Engineering, Nepti': 'https://scoea.org/',
    'Jawaharlal Darda Institute of Engineering and Technology, Yavatmal': 'https://jdiet.ac.in/',
    "Shri. Sai Shikshan Sanstha, Nagpur Institute of Technology, Nagpur": 'http://www.nit.edu.in/#',
    "Shree Gajanan Maharaj Shikshan Prasarak Manda'l Sharadchandra Pawar College of Engineering, Dumbarwadi": 'https://spcoe.in/',
    "Samridhi Sarwajanik Charitable Trust, Jhulelal Institute of Technology, Nagpur": 'https://www.jitnagpur.edu.in/',
    'R. C. Patel Institute of Technology, Shirpur': 'https://www.rcpit.ac.in/',
    'Sanmati Engineering College, Sawargaon Barde, Washim': 'https://sanmati.in/',
    "Dr. J. J. Magdum Charitable Trust's Dr. J.J. Magdum College of Engineering, Jaysingpur": 'https://www.jjmcoe.ac.in/',
    'Chhartrapati Shivaji Maharaj Institute of Technology, Shedung, Panvel': 'https://csmit.in/',
    "Shri.Someshwar Shikshan Prasarak Mandals, Someshwar Engineering College, Someshwar Nagar": 'https://www.secsomeshwar.ac.in/',
    "Gurunanak Educational Society's Gurunanak Institute of Technology, Nagpur": 'https://gnit.in/',
    "Nagaon Education Society's Gangamai College of Engineering, Nagaon, Tal Dist Dhule": 'https://gangamaiengg.org.in/',
    'Suman Ramesh Tulsiani Technical Campus: Faculty of Engineering, Kamshet,Pune.': 'https://www.srttc.ac.in/',
    "KSGBS's Bharat- Ratna Indira Gandhi College of Engineering, Kegaon, Solapur": 'https://bigce.in/',
    'Rajiv Gandhi College of Engineering Research & Technology Chandrapur': 'https://www.rcert.ac.in/',
    "Shree Santkrupa Shikshan Sanstha, Shree Santkrupa Institute Of Engineering & Technology, Karad": 'http://www.sietghogaon.org/',
    "N. B. Navale Sinhgad College of Engineering, Kegaon, Solapur": 'http://sinhgadsolapur.org/EdSite/',
    "Shree Yash Pratishthan, Shreeyash College of Engineering and Technology, Aurangabad": 'https://sycet.org/',
    'Sharad Institute of Technology College of Engineering, Yadrav(Ichalkaranji)': 'https://www.sitcoe.ac.in/',
    "Shahajirao Patil Vikas Pratishthan, S.B.Patil College of Engineering, Vangali, Tal. Indapur": 'https://www.sbpcoe.com/',
    'Aurangabad College of Engineering, Naygaon Savangi, Aurangabad': 'https://aurangabadengg.in/',
    'Universal College of Engineering,Kaman Dist. Palghar': 'https://universalcollegeofengineering.edu.in/',
    "Vishwabharati Academy's College of Engineering, Ahmednagar": 'http://vacoea.com/',
    "Anjuman College of Engineering & Technology, Nagpur": "https://www.anjumanengg.edu.in/",
    "Al-Ameen Educational and Medical Foundation, College of Engineering, Koregaon, Bhima": "https://bcud.unipune.ac.in/utilities/college_search/CEGP014640_ENG/Pune_University_College",
    "Sarvasiddhanta Education Soc's Nuva College of Engineering and Technology, Nagpur": "http://www.nuvaedu.com/",
    "Tatyasaheb Kore Institute of Engineering and Technology, Warananagar": "http://tkietwarana.ac.in/home.aspx",
    "Dhamangaon Education Society's College of Engineering and Technology, Dhamangaon": "https://collegedunia.com/college/13343-dhamangaon-education-societys-college-of-engineering-and-technology-amravati",
    "Maulana Mukhtar Ahmad Nadvi Technical Campus, Malegaon": "https://mmantc.edu.in/",
    "Dnyanshree Institute Engineering and Technology, Satara": "https://www.dnyanshree.edu.in/",
    "Gramodyogik Shikshan Mandal's Marathwada Institute of Technology, Aurangabad": "https://engg.mit.asia/",
    "Matsyodari Shikshan Sansatha's College of Engineering and Technology, Jalna": "https://www.msscetjalna.org/",
    "Nagnathappa Halge Engineering College, Parli, Beed": "https://www.nhce.in/",
    "Sahakar Maharshee Shankarrao Mohite Patil Institute of Technology & Research, Akluj": "https://www.smsmpitr.edu.in/",
    "Shri Gulabrao Deokar College of Engineering, Jalgaon": "https://www.sgdpjalgaon.org/",
    "Leela Education Society, G.V. Acharya Institute of Engineering and Technology, Shelu, Karjat": "https://gvaiet.org/",
    "Excelsior Education Society's K.C. College of Engineering and Management Studies and Research, Kopri, Thane (E)": "https://kccemsr.edu.in/",
    "Koti Vidya Charitable Trust's Smt. Alamuri Ratnamala Institute of Engineering and Technology, Sapgaon, Tal. Shahapur": "https://armiet.in/",
    "Haji Jamaluddin Thim Trust's Theem College of Engineering, At. Villege Betegaon, Boisar": "https://theemcoe.org/about-us.html",
    "Pradnya Niketan Education Society's Nagesh Karajagi Orchid College of Engineering & Technology, Solapur": "https://www.orchidengg.ac.in/",
    "Shetkari Shikshan Mandal's Pad. Vasantraodada Patil Institute of Technology, Budhgaon, Sangli": "https://pvpitsangli.edu.in/web/",
    "Amar Seva Mandal's Shree Govindrao Vanjari College of Engineering & Technology, Nagpur": "http://www.gwcet.ac.in/",
    "Shri Hanuman Vyayam Prasarak Mandals College of Engineering & Technology, Amravati": "http://hvpmcoet.in/",
    "Terna Public Charitable Trust's College of Engineering, Osmanabad": "https://coeosmanabad.ac.in/",
    "Shri. Vile Parle Kelavani Mandal's Institute of Technology, Dhule": "https://www.svkm-iot.ac.in/",
    "Shri. Balasaheb Mane Prasarak Mandals, Ashokrao Mane Group of Institutions, Kolhapur": "http://www.amgoi.org/",
    "Wainganga College of Engineering and Management, Dongargaon, Nagpur": "http://wcem.in/",
    "Shree L.R. Tiwari College of Engineering, Mira Road, Mumbai": "https://slrtce.in/",
    "Bharat College of Engineering, Kanhor, Badlapur(W)": "https://www.bharatenggcollege.com/",
    "Gharda Foundation's Gharda Institute of Technology,Khed, Ratnagiri": "https://www.git-india.edu.in/git/index.html",
    "Suryodaya College of Engineering & Technology, Nagpur": "https://scetngp.com/",
    "JMSS Shri Shankarprasad Agnihotri College of Engineering, Wardha": "https://sspace.ac.in/",
    "Marathwada Shikshan Prasarak Mandal's Shri Shivaji Institute of Engineering and Management Studies, Parbhani": "https://www.ssiems.org.in/",
    "Sir Shantilal Badjate Charitable Trust's S. B. Jain Institute of technology, Management & Research, Nagpur": "https://www.sbjit.edu.in/about-us/",
    "Datta Meghe Institute of Medical Science's Datta Meghe Institute of Engineering & Technology & Research, Savangi (Meghe)": "https://www.dmiher.edu.in/",
    "Shree Tuljabhavani College of Engineering, Tuljapur": "https://www.stbcet.org.in/",
    "B.R. Harne College of Engineering & Technology, Karav, Tal-Ambernath": "http://brharnetc.edu.in/br/",
    "Bhagwant Institute of Technology, Barshi": "https://bitbarshi.edu.in/index.html",
    "Shri. Ambabai Talim Sanstha's Sanjay Bhokare Group of Institutes, Miraj": "http://www.sbgimiraj.org/",
    "Rajendra Mane College of Engineering & Technology Ambav Deorukh": "https://rmcet.com/rmcet/index/v/",
    "Jai Mahakali Shikshan Sanstha, Agnihotri College of Engineering, Sindhi(Meghe)": "http://www.acenagthana.ac.in/",
    "T.M.E. Society's J.T.Mahajan College of Engineering, Faizpur": "https://www.jtmcoef.ac.in/",
    "S.S.P.M.'s College of Engineering, Kankavli": "https://sspmcoe.ac.in/",
    "Nanasaheb Mahadik College of Engineering,Walwa, Sangli": "https://www.nmcoe.org.in/",
    "P.S.G.V.P. Mandal's D.N. Patel College of Engineering, Shahada, Dist. Nandurbar": "https://www.coeshahada.ac.in/",
    "Jaidev Education Society, J D College of Engineering and Management, Nagpur": "https://jdcoem.ac.in/",
    "Anjuman-I-Islam's Kalsekar Technical Campus, Panvel": "https://aiktc.ac.in/",
    "Phaltan Education Society's College of Engineering Thakurki Tal- Phaltan Dist-Satara": "http://www.coephaltan.edu.in/",
    "Shri. Dadasaheb Gawai Charitable Trust's Dr. Smt. Kamaltai Gawai Institute of Engineering & Technology, Darapur, Amravati": "http://www.coephaltan.edu.in/",
    "Dilkap Research Institute Of Engineering and Management Studies, At.Mamdapur, Post- Neral, Tal- Karjat, Mumbai": "https://www.driems.in/",
    "Saraswati Education Society, Yadavrao Tasgaonkar College of Engineering and Management, Nasarapur, Chandai, Karjat": "https://ytcem.com/",
    "Mahatma Education Society's Pillai HOC College of Engineering & Technology, Tal. Khalapur. Dist. Raigad": "https://phcet.ac.in/",
    "M.D. Yergude Memorial Shikshan Prasarak Mandal's Shri Sai College of Engineering & Technology, Badravati": "https://sscet.in/",
    "Shri Shivaji Vidya Prasarak Sanstha's Late Bapusaheb Shivaji Rao Deore College of Engineering,Dhule": "https://ssvpsengg.ac.in/",
    "Holy-Wood Academy's Sanjeevan Engineering and Technology Institute, Panhala": "http://www.seti.edu.in/",
    "Samarth Education Trust's Arvind Gavali College Of Engineering Panwalewadi, Varye,Satara": "https://agce.edu.in/",
    "Dr.Rajendra Gode Institute of Technology & Research, Amravati": "https://www.drgitr.com/",
    "Shree Siddheshwar Women's College Of Engineering Solapur": "https://www.sswcoe.edu.in/",
    "Indala College Of Engineering, Bapsai Tal.Kalyan": "https://icoe.ac.in/",
    "Pune Vidyarthi Griha's College of Engineering and Technology, Pune": "https://www.pvgcoet.ac.in/",
    "Anjuman-I-Islam's M.H. Saboo Siddik College of Engineering, Byculla, Mumbai": "https://mhssce.ac.in/",
    "Rajgad Dnyanpeeth's Technical Campus,Dhangwadi, Bhor": "https://www.rajgad.edu.in/",
    "Jaywant College of Engineering & Management, Kille Macchindragad Tal. Walva": "http://www.sspmjcep.com/",
    "New Horizon Institute of Technology & Management, Thane": "https://nhitm.ac.in/",
}




class algorithms():
    
    def euclideanDistance(self,data1, data2, length):
        distance = 0
        for x in range(length):
            distance += np.square(data1[x] - data2[x])
        return np.sqrt(distance)                                                                                             


    def knn(self,trainingSet, testInstance, k):
        print(k)
        distances = {}
        sort = {}
        length = testInstance.shape[1]
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet.iloc[x], length)
            distances[x] = dist[0]
        sorted_d = sorted(distances.items(), key=lambda x: x[1])
        neighbors = []
        for x in range(k):
            neighbors.append(sorted_d[x][0])
        classVotes = {}
        for x in range(len(neighbors)):
            response = trainingSet.iloc[neighbors[x]][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        return (sortedVotes, neighbors)
    
    def getAccuracy(self,testSet,predictions):
        correct=0
        for x in range(len(testSet)):
            if testSet[x][-1] is predictions[x]:
                correct+=1
        return (correct/float(len(testSet)))*100.0
    
                    
    def predictSVM(self,kcet,caste):
        
        if caste=="GOBC":
            print("caste obc")
            data = pd.read_csv('cutoffobc.csv',encoding= 'unicode_escape')
            processed_data = data[['kcet','college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data)/5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college', axis=1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college',1)
            user_caste=str(caste)
            clf =svm.SVC()    
            clf.fit(train_input_data,train_output_data)
            marks=float(kcet)
            output_college1=clf.predict([[marks]])
            output_college=output_college1[0]
            return output_college
        
        elif caste=="GOPEN":
            print("caste open")
            data = pd.read_csv('cutofflist.csv',encoding= 'unicode_escape')
            processed_data = data[['kcet','college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data)/5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college',1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college',1)
            user_caste=str(caste)
            clf =svm.SVC()
            clf.fit(train_input_data,train_output_data)
            marks=float(kcet)
            output_college1=clf.predict([[marks]])
            output_college=output_college1[0]
            return output_college
    
    def predictKNN_with_links(self, kcet, caste):
        
        user_caste = str(caste)
        if caste == "GOBC":
            data = pd.read_csv('cutoffobc.csv', encoding='unicode_escape')
            processed_data = data[['kcet', 'college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data) / 5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college', axis=1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college',  axis=1)                          

        elif caste == "GOPEN":
            data = pd.read_csv('cutofflist.csv', encoding='unicode_escape')
            processed_data = data[['kcet', 'college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data) / 5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college', axis=1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college', axis=1)

        kcet = float(kcet) 
        testSet = [[kcet]]
        test = pd.DataFrame(testSet)
        k = 5
        result, neigh = self.knn(processed_data, test, k)
        list1 = []
        list2 = []
        for i in result:
            list1.append(i[0])
        for i in result:
            list2.append(i[1])
        # Predicted college name
        predicted_college = list1[0]

        # Get the URL from the dictionary
        college_url = college_links.get(predicted_college, 'Website not found')
        # cutoff_value = list2[0]
        # result_dict = {"name": predicted_college, "cutoff": cutoff_value, "url": college_url}

        return {"predicted_college": predicted_college, "college_url": college_url}

    
    def top_ten(self):

        data = pd.read_csv('cutofflist.csv',encoding= 'unicode_escape')
        my_list1=data.nlargest(10, ['kcet'])
        my_list=my_list1.values.tolist()
        return my_list
    
    def get_by_range(self,start,end,results,caste):

        res=results
        if caste=="GOPEN":
            data = pd.read_csv('cutofflist.csv',encoding= 'unicode_escape')
        elif caste=="GOBC":
            data = pd.read_csv('cutoffobc.csv',encoding= 'unicode_escape')
        
        my_data=data = data[(data['kcet'] >= start) & (data['kcet'] <= end)]
        l1=data.head(res)
        my_list=l1.values.tolist()  
        return my_list

c=algorithms()
testSet=[[99.964,'College of Engineering, Pune'],[99.164,'College of Engineering, Pune'],[99.964,'Pune Institute of Computer Technology, Dhankavdi, Pune'],[99.409,'Walchand College of Engineering, Sangli']]   
predictions=['College of Engineering, Pune','Pune Institute of Computer Technology, Dhankavdi, Pune','Pune Institute of Computer Technology, Dhankavdi, Pune','Walchand College of Engineering, Sangli']
accuracy=c.getAccuracy(testSet,predictions)
print("--------------------Accuracy--------")
print("Accuracy of knn is {}".format(accuracy))




