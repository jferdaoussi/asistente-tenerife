---
title: Asistente turístico de Tenerife
emoji: 🌋
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Guía de Tenerife con citas, tiempo y rutas (RAG + Gemini)
---

# Asistente turístico de Tenerife

Asistente conversacional que responde con la guía *TENERIFE.pdf* citando la
página, consulta el pronóstico del tiempo en Open-Meteo y da la ruta en
Google Maps. El panel lateral muestra, en cada turno, las herramientas
usadas, los tokens y la latencia.

Proyecto final de la asignatura *Large Language Models* (Máster en IA, Cloud
Computing y DevOps, Pontia). Demo con límites de uso: 20 mensajes por
conversación.
