//! SFCF binary format parsing helpers.
//!
//! Shared little-endian binary read utilities used by both
//! ``weight_loader`` and ``compute_graph`` for parsing the
//! embedded SFCF blob in compiled .dylib files.

pub fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, anyhow::Error> {
    if *pos >= data.len() {
        anyhow::bail!("truncated at pos {} (need u8)", pos);
    }
    let val = data[*pos];
    *pos += 1;
    Ok(val)
}

pub fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, anyhow::Error> {
    if *pos + 4 > data.len() {
        anyhow::bail!("truncated at pos {} (need u32)", pos);
    }
    let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(val)
}

pub fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, anyhow::Error> {
    if *pos + 8 > data.len() {
        anyhow::bail!("truncated at pos {} (need u64)", pos);
    }
    let val = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(val)
}

pub fn read_string(data: &[u8], pos: &mut usize) -> Result<String, anyhow::Error> {
    let len = read_u32(data, pos)? as usize;
    if *pos + len > data.len() {
        anyhow::bail!("truncated string at pos {} (need {} bytes)", pos, len);
    }
    let s = String::from_utf8(data[*pos..*pos + len].to_vec())?;
    *pos += len;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_u32_roundtrip() {
        let data = [0x78, 0x56, 0x34, 0x12];
        let mut pos = 0;
        assert_eq!(read_u32(&data, &mut pos).unwrap(), 0x12345678);
        assert_eq!(pos, 4);
    }

    #[test]
    fn test_read_u32_truncated() {
        let data = [0x01, 0x02];
        let mut pos = 0;
        assert!(read_u32(&data, &mut pos).is_err());
    }

    #[test]
    fn test_read_string_roundtrip() {
        let mut buf = Vec::new();
        let s = "hello";
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        let mut pos = 0;
        assert_eq!(read_string(&buf, &mut pos).unwrap(), "hello");
    }
}
