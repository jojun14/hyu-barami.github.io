---
title: 주식 가격 표시기
author: Jeonghyun Kim
date: 2022-11-15 15:30:00 +0900
categories: [Exhibition, 2022년]
tags: [post,jeonghyun,about-post]     # TAG names should always be lowercase, 띄어쓰기도 금지 
---

------------------------------------------
# 제작 동기 
최근 몇 년간 전례 없는 사회 환경으로 인해 주식시장에는 대 상승장과 대 하락장이 반복적으로 나타나고 있습니다. 이로 인한 각계 각층의 주식 투자 열풍도 상당한 수준입니다. (물론 최근 대 하락장 때문에 많이 꺽이긴 했습니다만, 열기만은 여전하다고 생각합니다.) 
주식 투자에 필수적인 주식 가격은 PC HTS나 웹, 스마트폰 MTS에서 쉽게 확인할 수 있지만 나스닥이나 한국 거래소 본관에 있는 주식 가격 표시 전광판을 MCU만을 이용해 직접 구현해보고 싶어 시작한 프로젝트입니다.
바라미 동아리실에 설치하면 주식 가격 표시는 물론 원격지에서 전송한 메시지를 표시하거나 동아리실 상주 인원 감지를 위한 단말기로 활용 등 다양한 추가 기능의 구현을 기대할 수도 있습니다.

# 아키텍처
위 사진은 본 작품의 아키텍처 (구조 설계) 요약도입니다. 크게 아래와 같이 3가지 요소로 구성됩니다.
<img src="/assets/img/post/2022-11-15-stock-display/architecture.jpg">

- **시세 서버** : 보통 증권사들은 한국 거래소, NYSE 같은 1차 가격 제공자들과 직접 협약을 맺어 전용 회선과 프로토콜로 정보를 받아옵니다. 
하지만 저희는 그럴 협상력은 없기 때문에 이러한 1차 제공자들로부터 가격정보를 받아와 최종 사용자가 보기 쉽게 가공하여 HTTP등의 형태로 제공하는 네이버 증권, 인베스팅 닷컴등의 2차 제공자에게서 가격 정보를 긁어옵니다. HTTP로 받아온 정보에는 단순히 가격이 바로 포함되있는 것이 아닌, 동적 렌더링과 통신이 포함되어 있어 임베디드 시스템에 바로 접속하여 사용하기엔 어렵니다. 그래서 본 작품에서는는 3차 제공자의 역할을 하는 시세 서버 별도로 구현했습니다.
- **MCU** : 사진에 STM32H745라고 표시되어 있는 요소입니다. 실제 작품에서는 반도체 수급 문제로 STM32H743 프로세서를 사용했습니다. Ethernet PHY + TCP/IP 스택을 구현하고 JSON으로 직렬화된 가격 메세지들을 MQTT 메세징 프로토콜을 통해 받아와 저장하고 간단한 그래픽 처리 후 DAC를 제어하는 역할을 맡습니다.
- **모니터** : 그래픽 DAC에서 생성한 VGA 비디오 신호를 받아 표시합니다. 일반 상용 모니터입니다.

# 특징과 작동 방식
- 작품의 주요 특징이자 특색으로는 단일 MCU만을 통한 주식 가격 크롤링 및 디스플레이 처리입니다. 라즈베리파이등 범용 OS가 올라간 MPU를 이용하면 동일 기능을 매우 쉽게 만들 수 있는 작품이나 그러한 오버헤드 없이 강인하고 가벼운 시스템을 구현하고자 했습니다. 
- 예를 들어 가격적인 측면에서 라즈베리파이 4 약 14만원에 비하여 본 작품 메인 보드 약 4만원 + DAC 보드 1만원이며 거의 1/3입니다. 소비 전력의 경우에도 라즈베리파이 약 3W, 본 작품 메인 보드 약 1W 수준으로 1/3입니다. 
- 펌웨어는 전원이 인가되면 Zephyr 커널 부팅이 완료되면 어플리케이션을 실행합니다. 어플리케이션은 자동으로 TCP/IPv4 스택을 시작하고 DHCPv4 프로토콜을 이용해 IP 주소를 할당받습니다. DNS 해석 및 WAN상의 MQTT 브로커에 연결하여 주식 가격 정보를 JSON 형식으로 받아옵니다. 최종적으로 이 가격 정보를 파싱하여 LVGL UI 라이브러리로 GUI 화면을 구성하고 VGA DAC를 통해 일반 모니터로 출력합니다.
- MQTT 브로커는 대표적인 주식 정보 제공 사이트 kr.investing.com에서 헤드리스 크롬 브라우저에 CDP(Chrome DevTools Protocol)로 연결하여 동적으로 렌더링된 DOM에서 가격 정보에 대한 정보를 추출해 일정 주기마다 메세지를 방송합니다.

# 세부 스펙과 기술 스택
- **메인 보드** : [ST Nucleo-H743ZI2 개발 보드](https://www.st.com/en/evaluation-tools/nucleo-h743zi.html); CPU ST STM32H743ZIT6 (Cortex-M7, LQFP144), 이더넷 PHY LAN8742A (10M/100M, RMII), RJ45 커넥터, STLINK-V3 디버거, USB 2.0 OTG FS
- **CPU** : [ST STM32H743ZIT6](https://www.st.com/en/microcontrollers-microprocessors/stm32h743zi.html); 32bit Cortex-M7, 최대 클럭 속도 480MHz (본 작품에서 96MHz 사용됨), FLASH 2MB, SRAM 1MB (약 4개의 섹터로 구분된, 본 작품에 512KB 섹터만 사용), LTDC 디스플레이 제어기 최대 XGA 해상도 및 색공간 RGB888 지원 (본 작품에서 VGA 해상도 및 RGB565 사용됨), 2계층의 프레임 버퍼와 DMA2D 지원, 하드웨어 JPEG 코덱, RMII 이더넷 지원, MPU 내장, 16KB L1 캐시
- **VGA DAC** : GM7123; (ADV7123의 중국판 호환품), CMOS 기반 최대 Max 330MHz (330MSPS) (본 작품에서 25MHz 사용됨), 10bits 3채널 고속 비디오 DAC, 최대 지원 해상도 및 주사율 1600x1200@100Hz (본 작품에서 640x480@60Hz 사용됨)
- **펌웨어** : Zephyr RTOS v3.1.0 (w/ TCP/IP, DNS, DHCP, MQTT, JSON, POSIX, DeviceTree subsys), LVGL UI 라이브러리
- **MQTT 브로커 (서버)** : Golang 1.19.와 헤드리스 크롬 브라우저, chromedp를 통해 CDP로 통신, MQTT 라이브러리, Docker 컨테이너로 구동

<img src="/assets/img/post/2022-11-15-stock-display/schemetic.png">
<img src="/assets/img/post/2022-11-15-stock-display/gerber.png">
<img src="/assets/img/post/2022-11-15-stock-display/assembly.png">

# 소스 코드
- 펌웨어 : https://github.com/Dictor/hangang-view-code
- 시세 서버 : https://github.com/Dictor/hangang-view-server
- 회로도와 거버 : https://github.com/Dictor/hangang-view-circuit

# 개선 필요점
- 모니터의 화면이 지글지글 끓는 듯이 불안정함을 확인할 수 있습니다. 보드 구조 상 CPU의 클럭 주파수가 Crystal에서 직접 생성되는 것이 아닌 그리 정밀하지 않은 디버거의 PLL에서 생성되어, 이 클럭을 분주하여 사용하는 Video DAC의 기준 클럭 주파수도 불안정해짐이 원인으로 추정됩니다.
- 이러한 노이즈를 잡기 위한 회로 및 보드 수정과 WIFI 스택 탑재 등 기능적 개선점이 추후 연구되어야 합니다.