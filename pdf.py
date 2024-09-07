from fpdf import FPDF


class pdf(FPDF):

    def __int__(self):
        self.newFile()

    def newFile(self, pathFile, mesiAnalizzati, mediaMax, valoreMediaMax, mediaMin, valoreMediaMin, RisultatiBorough):
        self.alias_nb_pages()
        self.add_page()
        # Begin with regular font
        self.set_font('Arial', '', 14)
        self.ln(15)
        self.write(5,
                   "Questo programma fornisce uno studio sulla frequenza con la quale i taxi vengono utilizzati a NY. Per lo svolgimento della nostra analisi utilizziamo i dati pubblici delle rotte dei Taxi a NYC disponibili su https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page.")
        self.ln(5)
        self.write(10, "Come da richiesta dell'operatore abbiamo analizzato i seguenti mesi:")
        self.ln(10)
        self.write(10, mesiAnalizzati)
        self.ln(10)
        self.write(10, "Il mese con il valore media massimo delle corse per tutta NY corrisponde a: ")
        self.ln(10)
        self.write(10, mediaMax + "  Con il valore: " + valoreMediaMax)
        self.ln(5)
        self.write(10, "Il minimo valore media delle corse per tutta NY corrisponde a: ")
        self.ln(10)
        self.write(10, mediaMin + "  Con il valore: " + valoreMediaMin)
        self.ln(5)
        self.write(10, "Valore visibile anche all'interno del grafico che segue")
        self.ln(5)
        self.image(pathFile + '/ConfrontoMesiNy.jpg', 10, 140, 120)
        self.ln(5)
        self.add_page()
        self.ln(20)
        self.write(10,
                   "Abbiamo inoltre deciso di effettuare la stessa analisi anche per ogni Borough presente a NY ottenendo i seguenti risultati:")
        for borough, value in RisultatiBorough.items():
            self.ln(5)
            self.write(10, borough)
            self.ln(5)
            for key, value in value.items():
                self.write(10, key)
                self.write(10," : ")
                self.write(10, value)
                self.ln(5)
            self.image(pathFile + '/' + borough +".jpg", 10, 130, 140)
            self.add_page()
            self.ln(20)
        self.ln(5)
        self.output(pathFile + '/OutputAnalisi.pdf', 'F')

    def header(self):
        # Logo
        self.image('data/img/Logo.jpg', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(50)
        # Title
        self.cell(80, 10, 'Analisi dei Taxi di NY', 1, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Pagina ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
