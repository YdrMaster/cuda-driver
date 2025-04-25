# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `Graph`, `GraphSpore` and `CaptureStream`;
- Add `MappedMem`, `MemProp`, `PhyMem`, `VirByte` and `VirMem` to use Device Virtual Memory;

### Changed

- Clone cccl when Toolkit version is below 12, otherwise, use build-in cccl;

## [0.0.0]

### Changed

- Upgrade Rust to 2024 edition;

[Unreleased]: https://github.com/YdrMaster/cuda-driver/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/YdrMaster/cuda-driver/releases/tag/v0.0.0
