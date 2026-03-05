use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use exif::{In, Reader, Tag, Value};
use serde::Serialize;

#[derive(Serialize)]
pub struct ExifData {
    pub date: Option<String>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
}

pub fn extract_exif(path: &Path) -> ExifData {
    let mut data = ExifData {
        date: None,
        lat: None,
        lon: None,
    };

    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return data,
    };
    let exif = match Reader::new().read_from_container(&mut BufReader::new(file)) {
        Ok(e) => e,
        Err(_) => return data,
    };

    // Date: DateTimeOriginal or DateTime -> "YYYY-MM-DD"
    let date_field = exif
        .get_field(Tag::DateTimeOriginal, In::PRIMARY)
        .or_else(|| exif.get_field(Tag::DateTime, In::PRIMARY));
    if let Some(field) = date_field {
        let s = field.display_value().to_string();
        if let Some(date_part) = parse_exif_date(&s) {
            data.date = Some(date_part);
        }
    }

    // GPS
    data.lat = read_gps_coord(&exif, Tag::GPSLatitude, Tag::GPSLatitudeRef);
    data.lon = read_gps_coord(&exif, Tag::GPSLongitude, Tag::GPSLongitudeRef);

    data
}

fn parse_exif_date(s: &str) -> Option<String> {
    let s = s.trim();
    if s.len() >= 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-' {
        return Some(s[..10].to_string());
    }
    if s.len() >= 10 && s.as_bytes()[4] == b':' && s.as_bytes()[7] == b':' {
        let date = format!("{}-{}-{}", &s[..4], &s[5..7], &s[8..10]);
        return Some(date);
    }
    None
}

fn read_gps_coord(exif: &exif::Exif, coord_tag: Tag, ref_tag: Tag) -> Option<f64> {
    let coord_field = exif.get_field(coord_tag, In::PRIMARY)?;
    let ref_field = exif.get_field(ref_tag, In::PRIMARY);

    let rationals = match &coord_field.value {
        Value::Rational(v) if v.len() >= 3 => v,
        _ => return None,
    };

    let d = rationals[0].to_f64();
    let m = rationals[1].to_f64();
    let s = rationals[2].to_f64();
    let mut decimal = d + m / 60.0 + s / 3600.0;

    if let Some(rf) = ref_field {
        let ref_str = rf.display_value().to_string();
        if ref_str.contains('S') || ref_str.contains('W') {
            decimal = -decimal;
        }
    }

    Some((decimal * 1_000_000.0).round() / 1_000_000.0)
}
