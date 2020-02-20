/* Org2XM v1.1
* Converts Org songs from Cave Story to XM modules.
* Implementation 2008-04-02 by Jan "Rrrola" Kadlec. Public Domain.
* Updates by third party.
*
* v1.0: 2008-04-02
*   -Original release
* v1.1: 2015-nov-14
*   -Added support for Org-03 files
* TODO:
*   -Add option to toggle non-linear volumes and pan
*
* Usage: "org2xm input.org"
*        "org2xm input.org ORGxxxyy.DAT" to specify .DATfile used (default 210EN)
*        "org2xm input.org ORGxxxyy.DAT c" for compatibility (but quality is worse)
*
* Credits:
* - Pixel Studios for making Cave Story and composing all the songs
* - Pete Mackay for providing details about the original Org player routine
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PACKED __attribute__((packed))

#define read(to, bytes) fread(to, 1, bytes, f)
#define write(from, bytes) fwrite(from, 1, bytes, g)

#define VOL 51  // default volume

//////////////////////////////////////////////////////////////////////// Input
//Original executable appears to have been compiled in Bloodshed's Dev-C++ 4.9.9.2,
//where the PACKED define probably worked properly.
//This is not the case with GCC 4.7.1, so I added some pragmas.
//We need these structures packed be cause we read from file one struct at a time.
#pragma pack(push,1)
struct OrgHeader
{
    uint8_t magic[6];
    uint16_t msPerBeat;  // 1..2000
    uint8_t measuresPerBar;
    uint8_t beatsPerMeasure;
    uint32_t loopStart;  // in beats
    uint32_t loopEnd;
} PACKED header;
#pragma pack(pop)

struct Note
{
    uint32_t start;
    uint8_t len;
    uint8_t key;  // 0(C-0)..96(B-7)
    uint8_t vol;  // 0..255
    uint8_t pan;  // 0..12
} *note[16];


///////////////////////////////////////////////////// Immediate representation
#pragma pack(push,1)
struct Instrument
{
    uint16_t freqShift;
    uint8_t sample;  // melody(voice 0..7) 0-99, drums(voice 8..15) 0-11
    uint8_t noLoop;
    uint16_t notes;

    uint8_t drum;
    uint8_t instrument;
    int8_t finetune;

    int8_t lastPan;  // encoding state
    uint8_t lastVol;
    uint8_t played;
} PACKED t[16];
#pragma pack(pop)

struct Track
{
    float freq;  // if (freq != 0), a new note starts
    uint8_t vol;      // 1..64; if (volume == 0), the note ends
    int8_t pan;      // -127=left..127=right
} *n[16];


uint8_t *pat[256], patTable[256]; int patLen[256]; int patterns;

int instruments;    // number of used instruments
int tracks;         // number of tracks
int barLen;         // pattern length
int rows, bars;     // song length
int loop;           // does the song loop?

int compatibility;  // reset instrument at each note?
int verorg;         //version of loaded song


/////////////////////////////////////////////////////////////////////// Output
#pragma pack(push,1)
struct XMHeader
{
    uint8_t id[17];
    uint8_t moduleName[20];
    uint8_t eof;
    uint8_t trackerName[20];
    uint16_t version;
    uint32_t headerSize;
    uint16_t songLength;  // in patterns
    uint16_t restartPosition;
    uint16_t channels;  // should be even, but we don't care ;)
    uint16_t patterns;
    uint16_t instruments;
    uint16_t flags;
    uint16_t tempo;
    uint16_t bpm;
    uint8_t patternOrder[256];
} PACKED xmh = {
    "Extended Module: ", "", 0x1A, "Org2XM by Rrrola    ", 0x104, 0x114
};


struct XMInstrument
{
    uint32_t size;
    uint8_t instrumentName[22];
    uint8_t zero;
    uint16_t samples;
    uint32_t sampleHeaderSize;
    uint8_t misc[230];
    uint32_t sampleLength;
    uint32_t loopStart;
    uint32_t loopLength;
    uint8_t volume;
    int8_t finetune;
    uint8_t type;
    uint8_t panning;
    uint8_t relativeKey;
    uint8_t reserved;
    uint8_t sampleName[22];
} PACKED smp = {
    0x107, "Melody-00", 0, 1, 0x28, {}, 256, 0, 256, VOL, 0, 1, 128, 48, 0, ""
};
#pragma pack(pop)

/////////////////////////////////////////////////// Drums and drum accessories

struct SoundBank
{
    uint8_t magic[6];
    uint8_t verbank;
    uint8_t verorg;
    uint8_t snumMelo;
    uint8_t snumDrum;
    uint16_t lenMelo;
    uint32_t *tblLenDrum;
    char (*tblNameDrum)[22];
    int8_t *melody;
    int8_t *drums;

    uint32_t *tblOffDrum;
    uint32_t lenAllDrm;
} sbank;

int SBKload(char *path)
{
    FILE *f;
    uint32_t i, j;
    uint32_t fSize;
    if (!(f = fopen(path, "rb"))) goto Err10;

    //file size, just in case
    fseek(f , 0 , SEEK_END);
    fSize = ftell(f);
    rewind(f);

    //read sample bank
    //header
    read(&sbank.magic, 6);
    if (memcmp(sbank.magic, "ORGBNK", 6)) goto Err11;

    //bank version
    read(&sbank.verbank, 1);
    //Organya song version this bank is intended for
    read(&sbank.verorg, 1);

    //number of melody samples
    read(&sbank.snumMelo, 1);
    //number of drums
    read(&sbank.snumDrum, 1);

    //length of each melody sample
    uint16_t tmp = 0;
    uint16_t x = 0;
    read(&x, 1); tmp = (tmp << 8) + x;
    read(&x, 1); tmp = (tmp << 8) + x;
    sbank.lenMelo = tmp;

    //drum sample length and offset tables
    sbank.tblLenDrum = malloc(sbank.snumDrum * sizeof(uint32_t));
    sbank.tblOffDrum = malloc(sbank.snumDrum * sizeof(uint32_t));
    sbank.lenAllDrm = 0;
    uint32_t off = 0;
    for(i = 0; i < sbank.snumDrum; i++)
    {
        uint32_t tmp = 0;
        uint32_t x = 0;
        read(&x, 1); tmp = (tmp << 8) + x;
        read(&x, 1); tmp = (tmp << 8) + x;
        read(&x, 1); tmp = (tmp << 8) + x;
        read(&x, 1); tmp = (tmp << 8) + x;
        sbank.tblLenDrum[i] = tmp;
        sbank.lenAllDrm += tmp;
        //offsets
        sbank.tblOffDrum[i] = off;
        off += tmp;
    }

    //drum sample names
    #define MAXSTR 23
    sbank.tblNameDrum = malloc(sbank.snumDrum * sizeof(char[MAXSTR]));
    for(i = 0; i < sbank.snumDrum; i++)
        for(j = 0; j < MAXSTR; j++)
        {
            char c = (char)fgetc(f);
            sbank.tblNameDrum[i][j] = j == MAXSTR-1 ? '\0' : c;
            if (c == '\0') break;
        }

    //melody waves
    sbank.melody = malloc(sbank.snumMelo * sbank.lenMelo * 1);
    read(sbank.melody, sbank.snumMelo * sbank.lenMelo * 1);

    //drum waves
    sbank.drums = malloc(sbank.lenAllDrm * 1);
    read(sbank.drums, sbank.lenAllDrm * 1);

    //delta encode samples
    int8_t *buf;
    //melody
    for (i = 0 ; i < sbank.snumMelo; i++)
    {
        buf = &sbank.melody[i * sbank.lenMelo];
        for (j = sbank.lenMelo - 1; j > 0; --j)
            buf[j] -= buf[j - 1];
    }
    //drums
    for (i = 0 ; i < sbank.snumDrum; i++)
    {
        buf = &sbank.drums[sbank.tblOffDrum[i]];
        for (j = sbank.tblLenDrum[i] - 1; j > 0; --j)
            buf[j] -= buf[j - 1];
        buf[0] ^= 0x80;
    }

    fclose(f);

    return 0;
    //errors
    Err11: printf("Invalid samplebank header!\n");  return 11;
    Err10: printf("Couldn't open samplebank!\n");   return 10;
}


////////////////////////////////////////////////////////// XM pattern encoding
// returns whatever is supposed to be encoded for (track i, row j)
// assumes that (i, j-1) has already been processed

int key, finetune;

uint8_t wKey, wInst, wVol, wFine, wPan, wPanVol, wSkip;

void encode(int i, int j)
{
    void resetPanVol(void)
    {
        wPan = (n[i][j].pan != (wInst ? 0 : t[i].lastPan));
        wVol = (n[i][j].vol != (wInst ? VOL : t[i].lastVol));
    }

    if (!j) t[i].lastPan = t[i].lastVol = 0;
    if (j%barLen == 0) t[i].played = 0;  // independent patterns: drop running fx

    wKey = wInst = wVol = wFine = wPan = wPanVol = 0;
    wSkip = (j==header.loopEnd-1 && i==tracks-1);

    // kill looping notes at loop start
    if (j == header.loopStart && !t[i].noLoop) t[i].lastVol = -1;

    if (n[i][j].freq)
    {
        wKey = 1;

        finetune = (log2(n[i][j].freq/8363)*12 - t[i].finetune/128. + t[i].drum*36)*8 + .5;
        key = (finetune+4) / 8;
        finetune -= key*8;
		finetune *= 2;

        wInst = !t[i].played; if (compatibility) wInst = 1;
        wFine = !!finetune;
        resetPanVol();

        // if panning is default, set instrument
        if (wPan && !n[i][j].pan)
        {
            wInst = 1;
            resetPanVol();
        }

        // if the volume is default and panning can't improve, set instrument
        if (wVol && n[i][j].vol==VOL && (n[i][j].pan!=t[i].lastPan || !n[i][j].pan))
        {
            wInst = 1;
            resetPanVol();
        }


        // if there's panning with another effect, try panning in volume column
        if (wPan && (wFine||wSkip) && !wVol)
        {
            wPan = 0;
            wPanVol = 1;
        }
    } else {
        if (n[i][j].vol != t[i].lastVol) wVol = 1;
        if (n[i][j].vol && n[i][j].pan != t[i].lastPan) wPan = 1;

        // write note endings
        if (wVol && !n[i][j].vol)
        {
            wVol = wPan = wFine = 0;
            if (!t[i].noLoop) {
                wKey = 1; key = 0x60;
            }

            t[i].lastVol = 0;
        }
    }

    if (wInst) {
        t[i].lastPan = 0; t[i].lastVol = VOL; t[i].played = 1;
    }

    if (wVol) {
        wPanVol = 0; t[i].lastVol = n[i][j].vol;
    }

    if (wPanVol || wPan) t[i].lastPan = n[i][j].pan;
    if (wSkip) wPan = wFine = 0;
    if (wPan) wFine = 0;
}



///////////////////////////////////////////////////////////////////////// Main

int main(int argc, char** argv)
{
    int i, j, k, l, m;
    FILE *f, *g;

    if (argc < 2) goto Err1;
    if (argc > 3) compatibility = 1;

    //Read bank file
    int ret = SBKload(argc < 3 ? "ORG210EN.DAT" : argv[2]);
    if (ret) return ret;

    if (!(f = fopen(argv[1], "rb"))) goto Err2;


    // Read the Org file

    read(&header, sizeof(struct OrgHeader));

    //Check if sounbank meets requirements of org file
    if        (memcmp(header.magic, "Org-02", 6)) {
        verorg = 2;
    } else if (memcmp(header.magic, "Org-03", 6)) {
        verorg = 3;
        //special case, CS compat mode
        //this bank has only CS drums
        if(sbank.verorg == 0xFF) goto Err6;
    } else {
        goto Err3;
    }
    if(sbank.verorg < verorg) goto Err6;


    for (i=0; i<16; ++i) read(&t[i], 6);

    for (i=0; i<16; ++i)
    {
        note[i] = malloc(t[i].notes * sizeof(struct Note));

        if (i>=8)
        {
            t[i].sample += 100;  // drum
            t[i].noLoop = 1;
            t[i].drum = 1;
        } else {
            t[i].drum = 0;
        }

        for (j=0; j<t[i].notes; ++j) read(&note[i][j].start, 4);
        for (j=0; j<t[i].notes; ++j) read(&note[i][j].key, 1);
        for (j=0; j<t[i].notes; ++j) read(&note[i][j].len, 1);
        for (j=0; j<t[i].notes; ++j) read(&note[i][j].vol, 1);
        for (j=0; j<t[i].notes; ++j) read(&note[i][j].pan, 1);

        for (j=0; j<t[i].notes; ++j)  // find last beat
        if (rows < note[i][j].start + note[i][j].len + 1)
        rows = note[i][j].start + note[i][j].len + 1;
    }

    barLen = header.measuresPerBar*header.beatsPerMeasure;

    if (header.loopStart < rows)
    {
        loop = 1;
        bars = header.loopEnd / barLen;  // loop: end the song right afer loopEnd
    } else {
        loop = 0;
        bars = (rows+barLen-1) / barLen;  // no loop: finish last bar
    }
    rows = bars*barLen;
    fclose(f);


    // Convert notes to tracks, find number of instruments and tracks
    for (i=tracks=0; i<16; ++i)
    {
        if (!t[i].notes) continue;

        n[tracks] = malloc(rows * sizeof(struct Track));
        memset(n[tracks], 0, rows * sizeof(struct Track));

        for (j=0; j<t[i].notes; ++j)
        {
            k = note[i][j].start; if (k >= rows) continue;

            int vol = note[i][j].vol;
            int pan = note[i][j].pan;

            vol = (vol==0xff ? VOL : (vol/255.)*56.5+8.499); //org minimum volume adjustment
            pan = (pan==0xff ? 0 : (pan-6)*127/6);

            // "new note" or "change note parameters"?
            if (note[i][j].key != 0xff)
            {
                //TODO: deal with doubling of freq shift per octave
                n[tracks][k].freq = t[i].freqShift - 1000;
                if (t[i].drum)
                n[tracks][k].freq += 800 * note[i][j].key + 100;
                else
                n[tracks][k].freq += 8363 * pow(2, note[i][j].key/12.);

                // non-looping instruments don't need "note off" commands
                if (t[i].noLoop) note[i][j].len = 16;

                // fill rows with note parameters
                do {
                    n[tracks][k].vol = vol;
                    n[tracks][k].pan = pan;
                } while (--note[i][j].len && ++k<rows && !(n[tracks][k].freq));

            } else {
                for ( ; n[tracks][k].vol && !(n[tracks][k].freq) && k<rows; ++k)
                {
                    if (note[i][j].vol != 0xff) n[tracks][k].vol = vol;
                    if (note[i][j].pan != 0xff) n[tracks][k].pan = pan;
                }
            }

            t[tracks] = t[i];  // squish instrument info
        }
        ++tracks;
        free(note[i]);
    }

    // Find the best bmp+tempo combination, bpm preferably around 125

    unsigned bestTempo=0, bestBPM=0, bestE=-1;

    for (i=1; i<64; ++i)
    {
        unsigned bpm = 2500*i/header.msPerBeat;
        if (bpm>15 && bpm<32767)
        {
            int e = abs(2500000*i/bpm - header.msPerBeat*1000);
            if (bestE>e || (bestE==e && abs(bestBPM-125)>abs(bpm-125)))
            {
				if (bpm<512 || bestBPM==0) {
					bestTempo = i;
					bestBPM = bpm;
					bestE = e;
				}
            }
        }
    }

    if (!bestTempo) {
		bestTempo = 1;
		bestBPM = 32767;
	}

    // Find best finetune: minimize frequency distortion and E5x usage

    for (i=0; i<tracks; ++i)
    {
        double bestE=1e30; uint8_t bestFinetune;

        for (k=-64; k<64; k++)  // try every possible finetune
        {
            double e = 0;

            t[i].finetune = k;

            // mimic pattern encoding to find whether finetune is available
            for (j=0; j<rows; j++)
            {
                encode(i, j);
                if (wKey && key!=0x60)
                {
                    if (!wFine) finetune = 0;  // can't use finetune on this note :-(

                    float logfreq = log2(8363) + (key + finetune/8. + t[i].finetune/128. - t[i].drum*36)/12.;
                    float d = log2(n[i][j].freq) - logfreq;
                    e += d*d + (finetune ? 1e-8 : 0);
                }
                if (e > bestE) goto nextk;  // break if already worse than the best
            }

            bestFinetune = k;
            bestE = e;

            nextk: continue;
        }

        t[i].finetune = bestFinetune;
    }

    // Join instruments with the same sample, loop type and finetune

    for (i=0; i<tracks; ++i) t[i].instrument = i;

    for (i=0; i<tracks; ++i) if (t[i].instrument == i)
    {
        for (j=i+1; j<tracks; ++j)
        {
            if (t[j].sample == t[i].sample && t[j].finetune == t[i].finetune &&
            t[j].noLoop == t[i].noLoop)
            t[j].instrument = i;
        }
    }

    for (instruments=0, i=0; i<tracks; ++i)  // renumber them sequentially
    {
        if (t[i].instrument == i) t[i].instrument = ++instruments;
        else t[i].instrument = t[t[i].instrument].instrument;
    }

    // Create XM patterns

    for (k=0; k<bars; ++k)
    {
        int len;
        uint8_t *buf = pat[k] = malloc(5*barLen*tracks+9);
        memset(buf, 0, 5*barLen*tracks+9);

        *(uint32_t*)&buf[0] = len = 9;
        *(uint16_t*)&buf[5] = barLen;

        for (j=k*barLen; j<(k+1)*barLen; ++j) for (i=0; i<tracks; ++i)
        {
            encode(i, j);

            uint8_t p = 0x80 | wKey | wInst*2 | (wVol||wPanVol)*4 | (wPan||wSkip||wFine)*24;
            if (p != 0x9F) buf[len++] = p;

            // key column
            if (wKey) buf[len++] = key+1;

            // instrument column
            if (wInst) buf[len++] = t[i].instrument;

            // volume column
            if (wVol)
                buf[len++] = 0x10 + n[i][j].vol;
            else if (wPanVol)
                buf[len++] = 0xC0 + (n[i][j].pan>0x77 ? 0xF : n[i][j].pan+0x88>>4);

            // effect column
            if (wSkip) {
                buf[len++] = 0xB; buf[len++] = header.loopStart / barLen;
            }

            else if (wPan) {
                buf[len++] = 8; buf[len++] = n[i][j].pan + 0x80;
            }

            else if (wFine) {
                buf[len++] = 0xE; buf[len++] = 0x58 + finetune;
            }
        }

        *(uint16_t*)&buf[7] = len-9;
        patLen[k] = len;
    }

    // Find duplicate patterns

    for (i=0; i<bars; ++i) patTable[i] = i;

    for (i=0; i<bars; ++i) if (patTable[i] == i)
    for (j=i+1; j<bars; ++j)
    if (patLen[i] == patLen[j] && !memcmp(pat[i], pat[j], patLen[i]))
    patTable[j] = i;

    for (patterns=0, i=0; i<bars; ++i)  // renumber them sequentially
    {
        if (patTable[i] == i) patTable[i] = patterns++;
        else patTable[i] = patTable[patTable[i]];
    }

    // Save XM header and patterns

    // "path/Name.org" -> "path/Name.xm"
    argv[1][strlen(argv[1])-3] = 'x';
    argv[1][strlen(argv[1])-2] = 'm';
    argv[1][strlen(argv[1])-1] = 0;
    if (!(g = fopen(argv[1], "wb"))) goto Err2;

    // "path/Name.xm" -> "Name"
    for (i=strlen(argv[1]); i>0 && argv[1][i-1]!='\\' && argv[1][i-1]!='/'; --i) ;

    argv[1][strlen(argv[1])-3] = 0;
    memcpy(xmh.moduleName, &argv[1][i], strlen(&argv[1][i])>20 ? 20 : strlen(&argv[1][i]));
    xmh.songLength = bars;
    xmh.restartPosition = header.loopStart / barLen;
    xmh.channels = tracks;
    xmh.patterns = patterns;
    xmh.instruments = instruments;
    xmh.flags = 1;
    xmh.tempo = bestTempo;
    xmh.bpm = bestBPM;
    memcpy(xmh.patternOrder, patTable, bars);
    write(&xmh, sizeof(struct XMHeader));

    for (k=0, i=0; i<bars; ++i) if (patTable[i] == k)
    {
        write(pat[i], patLen[i]);
        ++k;
    }

    // Save XM instruments and samples

    for (k=1, i=0; i<tracks; ++i) if (t[i].instrument == k)
    {
        int8_t *sbuf;

        sprintf(smp.sampleName, "samples/%03d.wav", t[i].sample);


        smp.loopStart = 0;
        smp.finetune = t[i].finetune;

        memset(smp.instrumentName, 0, 22);
        if (t[i].drum)
        {
            uint8_t dsmp = t[i].sample - 100;
            sbuf = &sbank.drums[sbank.tblOffDrum[dsmp]];

            smp.type = 0;
            smp.loopLength = 0;
            smp.sampleLength = sbank.tblLenDrum[dsmp];
            strcpy(smp.instrumentName, sbank.tblNameDrum[dsmp]);
            smp.relativeKey = 12;
        } else {
            sbuf = &sbank.melody[t[i].sample * sbank.lenMelo];

            smp.type = 1;
            smp.loopLength = sbank.lenMelo;
            smp.sampleLength = sbank.lenMelo;
            sprintf(smp.instrumentName,
            t[i].freqShift==1000 ? "Melody%02d" : "Melody%02d %+d Hz",
            t[i].sample, t[i].freqShift-1000);
            //smp.relativeKey = 36;
            smp.relativeKey = 48;
        }

        write(&smp, sizeof(struct XMInstrument));
        write(sbuf, smp.sampleLength);

        ++k;
    }



    // Cleanup, error messages

    fclose(g);
    for (i=0; i<tracks; ++i) free(n[i]);
    for (k=0; k<bars; ++k) free(pat[k]);
    //TODO, dealocate soundbank stuff

    return 0;

    //errors
    Err6: printf("Bank version smaller than org version!\n");                   return 6;
    Err5: printf("Speed out of XM range!\n");                                   return 5;
    Err3: printf("Invalid org header!\n");                                      return 3;
    Err2: printf("Couldn't open file \"%s\"!\n", argv[1]);                      return 2;
    Err1: printf("Usage: \"org2xm infile.org ORGxxxyy.DAT\"\n"
    "    OR \"org2xm infile.org ORGxxxyy.DAT c\" (for compatible output)");     return 1;
}