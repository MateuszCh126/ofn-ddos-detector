const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, LevelFormat, BorderStyle, WidthType,
  ShadingType, Header, Footer, PageNumber,
} = require("docx");

const ACCENT = "2E5E8C";
const border = { style: BorderStyle.SINGLE, size: 1, color: "C9D4DF" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

const p = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 160 },
    ...opts,
    children: [new TextRun({ text, ...(opts.run || {}) })],
  });

const bullet = (text, bold = null) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: bold
      ? [new TextRun({ text: bold + " — ", bold: true }), new TextRun(text)]
      : [new TextRun(text)],
  });

const step = (title, text) =>
  new Paragraph({
    numbering: { reference: "steps", level: 0 },
    spacing: { after: 120 },
    children: [new TextRun({ text: title + ". ", bold: true }), new TextRun(text)],
  });

const h1 = (text) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const h2 = (text) =>
  new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });

const scenarioRow = (name, what, alarm) =>
  new TableRow({
    children: [
      new TableCell({ borders, width: { size: 2600, type: WidthType.DXA }, margins: cellMargins,
        children: [p(name, { run: { bold: true }, spacing: { after: 0 } })] }),
      new TableCell({ borders, width: { size: 4426, type: WidthType.DXA }, margins: cellMargins,
        children: [p(what, { spacing: { after: 0 } })] }),
      new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, margins: cellMargins,
        children: [p(alarm, { spacing: { after: 0 } })] }),
    ],
  });

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 220, after: 140 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "steps",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Strona ", size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" }),
          ],
        })],
      }),
    },
    children: [
      new Paragraph({
        spacing: { after: 60 },
        children: [new TextRun({ text: "Jak działa detektor ataków DDoS oparty na liczbach OFN", bold: true, size: 40, color: ACCENT })],
      }),
      new Paragraph({
        spacing: { after: 280 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } },
        children: [new TextRun({ text: "Proste wytłumaczenie — bez wzorów, krok po kroku", italics: true, size: 24, color: "555555" })],
      }),

      h1("O co w ogóle chodzi?"),
      p("Atak DDoS polega na tym, że ktoś zalewa sieć ogromną ilością sztucznego ruchu, " +
        "żeby serwery przestały nadążać i prawdziwi użytkownicy nie mogli się połączyć. " +
        "Nasz system obserwuje ruch na wielu routerach jednocześnie i ma jedno zadanie: " +
        "jak najszybciej krzyknąć „to atak!”, ale jednocześnie nie panikować, " +
        "gdy ruch rośnie z naturalnych powodów (np. nagła popularność strony)."),
      p("Najprościej myśleć o nim jak o doświadczonym strażniku, który zna normalny rytm " +
        "każdego wejścia do budynku. Nie reaguje na pojedynczą większą grupę gości, " +
        "ale gdy przy wielu wejściach NARAZ robi się tłoczno i tłum cały czas gęstnieje — podnosi alarm."),

      h1("Co to jest liczba OFN?"),
      p("OFN (Skierowana Liczba Rozmyta) to sposób zapisania pomiaru, który zamiast jednej liczby " +
        "przechowuje trzy rzeczy naraz:"),
      bullet("ile mniej więcej wynosi anomalia (nie jeden punkt, tylko „rozmyty” zakres — bo pomiary zawsze mają szum),", "wartość"),
      bullet("jak bardzo jesteśmy jej pewni (wąski kształt = pewność, szeroki = niepewność),", "pewność"),
      bullet("czy ruch rośnie, czy maleje — to jest właśnie owo „skierowanie”, którego zwykłe liczby rozmyte nie mają.", "kierunek"),
      p("Dzięki kierunkowi system odróżnia router, na którym ruch właśnie narasta (podejrzane), " +
        "od routera, na którym ruch opada (uspokajające). Te dwa głosy można od siebie " +
        "uczciwie odjąć — matematyka OFN gwarantuje, że głos „za” i identyczny głos „przeciw” " +
        "dokładnie się znoszą."),

      h1("Jak system podejmuje decyzję — 6 kroków"),
      step("Pomiar", "Co krok czasu każdy router zgłasza, ile pakietów przez niego przeszło."),
      step("Porównanie z normą", "Dla każdego routera system pamięta ostatnią historię ruchu i liczy z niej " +
        "„typowy poziom” oraz „typowy rozrzut” (używa mediany, więc pojedyncze skoki nie psują normy). " +
        "Bieżący pomiar zamienia na wynik w stylu: „o ile odchyleń od normy jesteśmy powyżej?”."),
      step("Budowa liczby OFN", "Z ostatnich 4 pomiarów powstaje jedna liczba OFN: jej szerokość oddaje niepewność, " +
        "a kierunek bierze się z dopasowania prostej do okna — jeśli prosta wyraźnie pnie się w górę, " +
        "router „głosuje za atakiem”. Próg jest dobrany tak, by zwykły szum prawie nigdy " +
        "nie wyglądał jak trend."),
      step("Zliczenie głosów", "Liczby OFN wszystkich routerów są składane w jeden globalny obraz sieci: " +
        "głosy „rosnące” dodają podejrzenia, „malejące” je odejmują, a niezdecydowane liczą się " +
        "tylko w jednej czwartej. Wynik jest uśredniany, więc nie zależy od tego, czy sieć ma 10 czy 100 routerów."),
      step("Wyostrzenie", "Globalny rozmyty obraz zamieniany jest z powrotem na jedną liczbę — " +
        "„poziom podejrzenia”. Można o nim myśleć jak o średniej temperaturze niepokoju w całej sieci."),
      step("Alarm z histerezą", "Alarm włącza się dopiero, gdy podejrzenie przekracza próg przez 2 kroki z rzędu " +
        "I jednocześnie odpowiednio wiele routerów głosuje „za”. Gaśnie dopiero, gdy podejrzenie " +
        "spadnie wyraźnie niżej — też przez 2 kroki. Dzięki temu alarm nie miga jak zepsuta lampka."),

      h1("Skąd system zna swoje ustawienia?"),
      p("Progi alarmu, wagi poszczególnych routerów i długości serii dobiera automatycznie " +
        "algorytm genetyczny — procedura inspirowana ewolucją. Tworzy populację kilkudziesięciu " +
        "losowych zestawów ustawień, sprawdza każdy na scenariuszach treningowych, najlepsze " +
        "„krzyżuje” i lekko „mutuje”, a po kilkudziesięciu pokoleniach zostaje zestaw, który " +
        "najlepiej godzi trzy cele: wykrywać jak najwięcej (55% wagi oceny), nie alarmować fałszywie (30%) " +
        "i reagować jak najszybciej (15%)."),

      h1("Na czym system jest testowany?"),
      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2600, 4426, 2000],
        rows: [
          new TableRow({
            children: ["Scenariusz", "Co się dzieje", "Czy ma być alarm?"].map((t, i) =>
              new TableCell({
                borders, margins: cellMargins,
                width: { size: [2600, 4426, 2000][i], type: WidthType.DXA },
                shading: { fill: "D9E4EF", type: ShadingType.CLEAR },
                children: [p(t, { run: { bold: true }, spacing: { after: 0 } })],
              })),
          }),
          scenarioRow("normal", "Zwykły ruch, tylko szum", "NIE"),
          scenarioRow("ddos_ramp", "Atak narastający liniowo na 70% routerów", "TAK"),
          scenarioRow("ddos_pulse", "Atak pulsujący (włącz/wyłącz)", "TAK"),
          scenarioRow("ddos_low_and_slow", "Atak słaby i powolny — najtrudniejszy", "TAK"),
          scenarioRow("ddos_rotating", "Atakujący przeskakuje między grupami routerów", "TAK"),
          scenarioRow("flash_crowd", "Nagła, ale uczciwa popularność (tłum prawdziwych użytkowników)", "NIE — pułapka!"),
          scenarioRow("flash_cascade", "Fala uczciwej popularności przesuwająca się po sieci", "NIE — pułapka!"),
        ],
      }),
      p("", { spacing: { after: 80 } }),
      p("Scenariusze „pułapki” są najważniejsze: łatwo zbudować system, który krzyczy przy każdym " +
        "wzroście ruchu. Sztuką jest milczeć, gdy wzrost jest uczciwy — i tu właśnie pomaga " +
        "kierunkowość OFN oraz wymaganie, by wiele routerów głosowało naraz."),

      h1("Co warto zapamiętać"),
      bullet("System nie patrzy na jeden licznik, tylko na zgodność wielu świadków — routerów."),
      bullet("Każdy świadek mówi nie tylko „ile”, ale też „w którą stronę to zmierza”."),
      bullet("Norma liczona jest medianą, więc atak nie jest w stanie szybko „przestawić” pojęcia normalności."),
      bullet("Alarm ma histerezę — włącza się i gaśnie z rozmysłem, nie miga."),
      bullet("Ustawienia stroi ewolucja (algorytm genetyczny), a nie człowiek zgadujący progi."),
      bullet("Najtrudniejszy przeciwnik to atak „low and slow” — celowo wolny, by udawać naturalny wzrost; " +
        "do jego niezawodnego łapania potrzebne jest strojenie pod konkretną sieć."),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("D:\\github\\github\\ofn-ddos-detector\\wytlumaczenie.docx", buffer);
  console.log("OK wytlumaczenie.docx");
});
