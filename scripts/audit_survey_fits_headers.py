import json
import re
import ssl
import urllib.request
import zlib

from astropy.io import fits

items = [
    "ges_dr5.1.fits",
    "rave_dr6_madera.fits",
    "rave_dr6_x_gaiaedr3.fits",
    "rave_dr6_cnn.fits",
    "rave_dr6_gaia_bdasp.fits",
    "astraMWMLite-0.6.0.fits.gz",
    "lamost-param/dr8_v2_LRS_stellar.fits",
    "lamost-param/dr8_v2_MRS_stellar.fits",
]
context = ssl.create_default_context()
context.load_cert_chain("/Users/fabbros/.ssl/cadcproxy.pem")
base = "https://ws-uv.canfar.net/arc/files/projects/k-pop/catalogues/"
rows = []
for name in items:
    try:
        req = urllib.request.Request(base + name, headers={"Range": "bytes=0-524287"})
        with urllib.request.urlopen(req, context=context, timeout=40) as response:
            status = response.status
            if status not in (200, 206):
                rows.append({"name": name, "status": status})
                continue
            # Some ARC servers ignore Range. Read only a bounded prefix and
            # close the stream; do not download the catalogue.
            data = response.read(524288)
        compressed = data[:2] == b"\x1f\x8b"
        if compressed:
            data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(
                data, 16 * 1024 * 1024
            )
        start = 0
        headers = []
        for h in range(2):
            end = None
            for offset in range(start, len(data), 80):
                if data[offset : offset + 8] == b"END     ":
                    end = offset + 80
                    break
            if end is None:
                raise ValueError("header exceeds bounded prefix")
            hdr = fits.Header.fromstring(data[start:end].decode("ascii"), sep="")
            padded = ((end - start + 2879) // 2880) * 2880
            columns = []
            for i in range(1, int(hdr.get("TFIELDS", 0)) + 1):
                n = str(hdr.get("TTYPE" + str(i), ""))
                if re.search(
                    r"(?i)(source.?id|gaia|object|obs.?id|^id|^cname|ra|dec|flag|warn|snr|teff|logg|feh|fe_h|m_h|\[fe|alpha|mg_fe|si_fe|ca_fe|ti_fe|^(e_)?(mg|si|ca|ti)[12]$)",
                    n,
                ):
                    columns.append(
                        {
                            "name": n,
                            "format": hdr.get("TFORM" + str(i)),
                            "unit": hdr.get("TUNIT" + str(i)),
                            "null": hdr.get("TNULL" + str(i)),
                        }
                    )
            headers.append(
                {
                    "hdu": h,
                    "extname": str(hdr.get("EXTNAME", "")),
                    "naxis1": hdr.get("NAXIS1"),
                    "naxis2": hdr.get("NAXIS2"),
                    "tfields": hdr.get("TFIELDS"),
                    "relevant_columns": columns,
                }
            )
            naxis = int(hdr.get("NAXIS", 0))
            ndata = 0 if naxis == 0 else abs(int(hdr["BITPIX"])) // 8
            for j in range(1, naxis + 1):
                ndata *= int(hdr.get("NAXIS" + str(j), 0))
            ndata = (ndata + int(hdr.get("PCOUNT", 0))) * int(hdr.get("GCOUNT", 1))
            start += padded + ((ndata + 2879) // 2880) * 2880
            if h == 0 and start >= len(data):
                raise ValueError("extension header not in bounded prefix")
        rows.append(
            {
                "name": name,
                "range_supported": status == 206,
                "compressed": compressed,
                "prefix_bytes_limit": 524288,
                "headers": headers,
            }
        )
    except Exception as e:
        rows.append(
            {"name": name, "error_type": type(e).__name__, "error": str(e)[:250]}
        )
print(json.dumps(rows, indent=2))
