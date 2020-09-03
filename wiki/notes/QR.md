---
attachments: [Clipboard_2020-08-22-01-25-28.png]
title: QR
created: '2020-08-21T23:02:24.715Z'
modified: '2020-08-21T23:26:28.620Z'
---

# QR

## Formats of QR Code:
- QR code with full range 
- micro QR code with really small capacity

## Encodable charset: 
- digits 0-9
- uppercase letters A-Z
- space % $ + * - . / :
- ISO/IEC 8859-1
- kanji characters 

## Representation of data:
- dark module for binary 1
- light module for binary 0
- could also be reversed

## Symbol size:
- 11 x 11 modules to 17 x 17 [micro QR - version M1 to M4 | increasing in steps of 2 modules]
- 21 x 21 modules to 177 x 177 modules [QR - version 1 to 40 | increasing in steps of 4 modules]

## Error correction:
- L 7%
- M 15%
- Q 25%
- H 30%

---
## Additional features:
- __Structured append__: allow using up to 16 QR codes to be represented in it. 
- __Extended Channel Interpretations__: allow using charset other than defined.
- __Reflectance reversal__: allow switch representation of data.
- __Mirror imaging__: allow decode with mirrored QR, useful on black backgrounds.

---
## Symbol Structure
![](@attachment/Clipboard_2020-08-22-01-25-28.png)
