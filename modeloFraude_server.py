#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 18:17:42 2025
@author: brunoalcantar
"""
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
import numpy as np

# Cargar el modelo y las features
with open('modelo_lightgbm_ensemble_con_features.pkl', 'rb') as f:
    final_model, features = pickle.load(f)

# Crear la app Flask
app = Flask(__name__)

@app.route('/inspect', methods=['GET'])
def inspect_model():
    """Endpoint temporal para inspeccionar el modelo"""
    try:
        info = {
            'model_type': str(type(final_model)),
            'features_saved': features,
            'features_count': len(features)
        }
        
        # Intentar obtener feature names del modelo
        if hasattr(final_model, 'feature_names_in_'):
            info['model_feature_names'] = list(final_model.feature_names_in_)
        
        # Si es pipeline
        if hasattr(final_model, 'steps'):
            info['pipeline_steps'] = []
            for name, step in final_model.steps:
                step_info = {'name': name, 'type': str(type(step))}
                if hasattr(step, 'feature_names_in_'):
                    step_info['feature_names'] = list(step.feature_names_in_)
                info['pipeline_steps'].append(step_info)
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # ========== GUARDAR VARIABLES ORIGINALES ANTES DEL RENOMBRADO ==========
        ingreso_original = data.get('income', 0)
        credito_solicitado = data.get('credito_solicitado', 0)  # NUEVA VARIABLE
        
        # Validar que credito_solicitado existe y es válido
        if credito_solicitado <= 0:
            return jsonify({'error': 'El campo credito_solicitado es requerido y debe ser mayor a 0'}), 400
        
        new_data = pd.DataFrame([data])
        
        # ---------- RENOMBRAR COLUMNAS ----------
        new_data.rename(columns={
            "employment_status": "employment_status",
            "housing_status": "housing_status"
        }, inplace=True)
        
        # Columnas después del rename (debug silencioso)

        # ---------- PREPROCESAMIENTO MEJORADO ----------
        
        # 1. EMPLOYMENT STATUS - Más granular y con mayor peso
        employment_status = new_data['employment_status'].iloc[0]
        if employment_status in ['CD', 'CE']:  # Empleado de tiempo completo/medio tiempo
            employment_score = 0.9
        elif employment_status == 'CF':  # Trabajador independiente
            employment_score = 0.6
        elif employment_status in ['CG', 'CH']:  # Pensionado/Estudiante
            employment_score = 0.4
        else:  # Desempleado u otros
            employment_score = 0.1
        
        new_data['employment_dummy'] = employment_score
        
        # 2. HOUSING STATUS - Más opciones y peso
        housing_status = new_data['housing_status'].iloc[0]
        if housing_status == 'BA':  # Casa propia
            housing_score = 0.9
        elif housing_status in ['BB', 'BC']:  # Hipoteca/Financiada
            housing_score = 0.7
        elif housing_status == 'BD':  # Rentada
            housing_score = 0.4
        else:  # Vive con familiares u otros
            housing_score = 0.2
        
        new_data['housing_dummy'] = housing_score
        
        # 3. SOURCE - Más diferenciación
        source = new_data['source'].iloc[0]
        if source == 'INTERNET':  # Solicitud online (más control)
            source_score = 0.8
        elif source == 'TELEAPP':  # Teléfono
            source_score = 0.6
        elif source == 'REFERRAL':  # Referido
            source_score = 0.7
        else:  # Otros canales
            source_score = 0.4
            
        new_data['source_encoded'] = source_score
        
        # 4. CUSTOMER AGE - Curva de riesgo más realista
        age = new_data['customer_age'].iloc[0]
        age = max(age, 18)  # Mínimo 18 años
        
        if age < 25:  # Muy joven - alto riesgo
            age_score = 0.3
        elif age < 35:  # Joven adulto - riesgo medio-alto
            age_score = 0.6
        elif age < 50:  # Adulto maduro - bajo riesgo
            age_score = 0.9
        elif age < 65:  # Pre-jubilación - riesgo medio
            age_score = 0.7
        else:  # Adulto mayor - riesgo alto
            age_score = 0.4
            
        new_data['customer_age'] = age_score
        
        # 5. BANK MONTHS COUNT - Relación más fuerte con la lealtad
        bank_months = new_data['bank_months_count'].iloc[0]
        bank_months = max(bank_months, 0)  # No negativos
        
        if bank_months >= 24:  # 2+ años - excelente
            bank_score = 0.9
        elif bank_months >= 12:  # 1+ año - bueno
            bank_score = 0.7
        elif bank_months >= 6:  # 6+ meses - regular
            bank_score = 0.5
        elif bank_months >= 3:  # 3+ meses - nuevo pero algo de tiempo
            bank_score = 0.3
        else:  # Muy nuevo - riesgo alto
            bank_score = 0.1
            
        new_data['bank_months_count'] = bank_score

        # 6. ESCALADO DE INGRESO MEJORADO (usando tu función que funcionó bien)
        def scale_income_mx(income, min_real=100, max_real=150000, min_scaled=0.01, max_scaled=0.99):
            """Escalado de ingreso mejorado con logs"""
            original_income = income
            income = max(min_real, min(income, max_real))
            scaled = ((income - min_real) / (max_real - min_real)) * (max_scaled - min_scaled) + min_scaled
            return scaled

        new_data['income'] = new_data['income'].apply(scale_income_mx)
        
        # 7. VARIABLES BINARIAS - Darles más peso en la lógica
        email_free = new_data['email_is_free'].iloc[0]
        has_cards = new_data['has_other_cards'].iloc[0]
        phone_home = new_data['phone_home_valid'].iloc[0]
        phone_mobile = new_data['phone_mobile_valid'].iloc[0]
        
        # Email gratuito = más riesgo
        new_data['email_is_free'] = 1 - email_free  # Invertir: email gratuito = mayor riesgo
        
        # Tener otras tarjetas puede ser bueno (experiencia) o malo (sobreendeudamiento)
        # Lo tratamos como positivo si el ingreso es alto, negativo si es bajo
        income_percentile = new_data['income'].iloc[0]
        if income_percentile > 0.6:  # Ingreso alto
            new_data['has_other_cards'] = has_cards * 0.8  # Positivo pero moderado
        else:  # Ingreso bajo-medio
            new_data['has_other_cards'] = (1 - has_cards) * 0.6  # No tener es mejor
        
        # Teléfonos válidos = menor riesgo
        new_data['phone_home_valid'] = phone_home
        new_data['phone_mobile_valid'] = phone_mobile

        # Debug features (silencioso)
        # print(f"📌 FEATURES ESPERADOS: {features}")
        # print(f"🟢 COLUMNAS DISPONIBLES: {list(new_data.columns)}")
        # print(f"💰 INGRESO ORIGINAL: {ingreso_original}")
        # print(f"💳 CRÉDITO SOLICITADO: {credito_solicitado}")

        # ---------- CALCULAR RISK SCORE MEJORADO ----------
        # En lugar de solo usar el modelo, combinamos modelo + lógica de negocio
        
        # Primero, intentar usar el modelo original
        orders_to_try = [
            ("Alfabético", [
                'bank_months_count', 'customer_age', 'email_is_free', 
                'employment_dummy', 'has_other_cards', 'housing_dummy',
                'income', 'phone_home_valid', 'phone_mobile_valid', 'source_encoded'
            ]),
            ("Originales primero", [
                'email_is_free', 'has_other_cards', 'phone_home_valid', 'phone_mobile_valid',
                'income', 'customer_age', 'bank_months_count',
                'employment_dummy', 'housing_dummy', 'source_encoded'
            ]),
            ("Como DataFrame", [
                'income', 'customer_age', 'email_is_free', 'has_other_cards', 
                'phone_home_valid', 'bank_months_count', 'phone_mobile_valid',
                'employment_dummy', 'housing_dummy', 'source_encoded'
            ])
        ]
        
        model_prediction = None
        successful_order = None
        
        for order_name, feature_order in orders_to_try:
            try:
                temp_input = pd.DataFrame()
                for feature in feature_order:
                    if feature in new_data.columns:
                        temp_input[feature] = new_data[feature]
                    else:
                        print(f"⚠️ Feature faltante en {order_name}: {feature} — rellenando con 0.5")
                        temp_input[feature] = 0.5
                
                # Intentar predicción silenciosa
                test_pred = final_model.predict_proba(temp_input)
                model_prediction = test_pred[0][1]  # Probabilidad de riesgo
                successful_order = order_name
                break
                
            except Exception as e:
                continue
        
        # Si el modelo falla, usar solo lógica de negocio
        if model_prediction is None:
            model_prediction = 0.5  # Neutro
        
        # ---------- COMBINAR MODELO + LÓGICA DE NEGOCIO ----------
        
        # Factores de riesgo basados en reglas de negocio
        income_factor = 1 - income_percentile  # Menor ingreso = mayor riesgo
        employment_factor = 1 - employment_score  # Peor empleo = mayor riesgo
        housing_factor = 1 - housing_score  # Peor vivienda = mayor riesgo
        age_factor = 1 - age_score  # Edad riesgosa = mayor riesgo
        bank_factor = 1 - bank_score  # Menos tiempo en banco = mayor riesgo
        
        # Variables binarias
        email_factor = email_free * 0.1  # Email gratuito añade riesgo
        phone_factor = (2 - phone_home - phone_mobile) * 0.05  # Teléfonos inválidos añaden riesgo
        
        # ---------- BONIFICACIONES PARA CLIENTES DE BAJO RIESGO ----------
        # Identificar perfiles excelentes y darles bonificaciones significativas
        
        low_risk_bonus = 0.0
        
        # Bonificación por perfil de ingreso excelente
        if income_percentile >= 0.8:  # Top 20% de ingresos
            low_risk_bonus += 0.15
        elif income_percentile >= 0.6:  # Top 40% de ingresos
            low_risk_bonus += 0.08
        
        # Bonificación por empleo estable de alta calidad
        if employment_score >= 0.9:  # Empleado tiempo completo
            low_risk_bonus += 0.12
        elif employment_score >= 0.6:  # Trabajo independiente estable
            low_risk_bonus += 0.06
        
        # Super bonificación por vivienda propia
        if housing_score >= 0.9:  # Casa propia
            low_risk_bonus += 0.10
        elif housing_score >= 0.7:  # Hipoteca (también es positivo)
            low_risk_bonus += 0.05
        
        # Bonificación por edad óptima
        if age_score >= 0.9:  # Edad óptima (35-50)
            low_risk_bonus += 0.08
        elif age_score >= 0.6:  # Edad buena
            low_risk_bonus += 0.04
        
        # Bonificación por relación bancaria larga
        if bank_score >= 0.9:  # 2+ años en el banco
            low_risk_bonus += 0.12
        elif bank_score >= 0.7:  # 1+ año en el banco
            low_risk_bonus += 0.07
        
        # Bonificación por perfil digital/profesional
        if new_data['email_is_free'].iloc[0] == 0:  # Email corporativo (no gratuito)
            low_risk_bonus += 0.06
        
        # Bonificación por validación completa de contacto
        if phone_home == 1 and phone_mobile == 1:
            low_risk_bonus += 0.05
        
        # Bonificación por canal confiable
        if source_score >= 0.8:  # Internet o referido
            low_risk_bonus += 0.04
        
        # SUPER BONIFICACIÓN PARA PERFILES PREMIUM
        premium_criteria = 0
        if income_percentile >= 0.7: premium_criteria += 1
        if employment_score >= 0.8: premium_criteria += 1
        if housing_score >= 0.8: premium_criteria += 1
        if age_score >= 0.8: premium_criteria += 1
        if bank_score >= 0.7: premium_criteria += 1
        
        if premium_criteria >= 4:  # Cumple 4+ criterios premium
            low_risk_bonus += 0.20
        elif premium_criteria >= 3:  # Cumple 3 criterios premium
            low_risk_bonus += 0.10
        
        print(f"📈 TOTAL BONIFICACIONES BAJO RIESGO: -{low_risk_bonus:.3f}")
        
        # Combinar factores con pesos
        business_risk = (
            income_factor * 0.35 +      # Ingreso es el factor más importante
            employment_factor * 0.25 +   # Empleo segundo más importante
            housing_factor * 0.15 +      # Vivienda importante
            age_factor * 0.10 +          # Edad moderadamente importante
            bank_factor * 0.10 +         # Relación bancaria
            email_factor +               # Factores menores
            phone_factor
        ) - low_risk_bonus              # Restar bonificaciones (reduce el riesgo)
        
        # Combinar predicción del modelo con lógica de negocio
        final_risk_score = (model_prediction * 0.4) + (business_risk * 0.6)
        
        # Asegurar que esté en rango [0, 1] pero permitir que llegue muy bajo
        final_risk_score = max(0.001, min(1, final_risk_score))  # Mínimo 0.001 en lugar de 0
        
        # Debug final silencioso (solo total de bonificaciones)
        # print(f"🎯 Riesgo final: {final_risk_score:.4f}, Bonificaciones: -{low_risk_bonus:.3f}")
        
        # ========== USAR EL CRÉDITO SOLICITADO PARA CALCULAR APROBACIÓN ==========
        credito_aprobado = calcular_credito_aprobado(final_risk_score, credito_solicitado, ingreso_original)
        
        return jsonify({
            'credit_risk_score': float(final_risk_score),
            'credito_solicitado': float(credito_solicitado),
            'credito_aprobado': float(credito_aprobado),
            'porcentaje_aprobado': float((credito_aprobado / credito_solicitado * 100) if credito_solicitado > 0 else 0),
            'detalles': {
                'prediccion_modelo': float(model_prediction) if model_prediction else None,
                'riesgo_logica_negocio': float(business_risk),
                'bonificaciones_aplicadas': float(low_risk_bonus),
                'criterios_premium_cumplidos': int(premium_criteria),
                'clasificacion_riesgo': clasificar_riesgo(final_risk_score),
                'decision_credito': obtener_decision_credito(final_risk_score),
                'factores_riesgo': {
                    'ingreso': float(income_factor),
                    'empleo': float(employment_factor),
                    'vivienda': float(housing_factor),
                    'edad': float(age_factor),
                    'banco': float(bank_factor)
                }
            }
        })
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

def clasificar_riesgo(risk_score):
    """
    Clasifica el nivel de riesgo basado en el score
    """
    if risk_score <= 0.05:
        return "EXCELENTE (Ultra Bajo Riesgo)"
    elif risk_score <= 0.15:
        return "MUY BUENO (Bajo Riesgo)"
    elif risk_score <= 0.30:
        return "BUENO (Riesgo Controlado)"
    elif risk_score <= 0.45:
        return "REGULAR (Riesgo Moderado)"
    elif risk_score <= 0.60:
        return "MALO (Alto Riesgo)"
    elif risk_score <= 0.75:
        return "MUY MALO (Muy Alto Riesgo)"
    else:
        return "CRÍTICO (Riesgo Extremo)"

def obtener_decision_credito(risk_score):
    """
    Obtiene la decisión de crédito basada en el riesgo
    """
    if risk_score > 0.70:
        return "RECHAZADO - Riesgo demasiado alto"
    elif risk_score > 0.50:
        return "APROBADO PARCIAL - Alto riesgo, crédito limitado"
    elif risk_score > 0.30:
        return "APROBADO - Riesgo moderado"
    elif risk_score > 0.15:
        return "APROBADO - Bajo riesgo, buenas condiciones"
    else:
        return "PRE-APROBADO - Excelente perfil, mejores condiciones"

def calcular_credito_aprobado(risk_score, credito_solicitado, ingreso):
    """
    Nueva lógica para calcular crédito aprobado basado en el crédito solicitado
    y penalizando fuertemente los casos de alto riesgo
    """
    
    # REGLA 1: Riesgo > 0.70 = SIN CRÉDITO
    if risk_score > 0.70:
        return 0.0
    
    # REGLA 2: Riesgo > 0.50 = MÁXIMO 20% del solicitado
    elif risk_score > 0.50:
        factor = min(0.20, (0.70 - risk_score) / 0.20 * 0.20)  # Escala del 0% al 20%
        credito_aprobado = credito_solicitado * factor
        return credito_aprobado
    
    # REGLA 3: Para riesgo <= 0.50, aplicar lógica progresiva más generosa
    else:
        # Escala progresiva basada en riesgo
        if risk_score <= 0.05:  # Ultra bajo riesgo
            factor = 1.50  # 150% del solicitado (oferta premium)
        elif risk_score <= 0.10:  # Muy bajo riesgo
            factor = 1.25  # 125% del solicitado
        elif risk_score <= 0.20:  # Bajo riesgo
            factor = 1.10  # 110% del solicitado
        elif risk_score <= 0.30:  # Riesgo controlado
            factor = 1.00  # 100% del solicitado
        elif risk_score <= 0.40:  # Riesgo moderado
            factor = 0.80  # 80% del solicitado
        else:  # risk_score <= 0.50 - Riesgo moderado-alto
            factor = 0.60  # 60% del solicitado
        
        credito_base = credito_solicitado * factor
        
        # VALIDACIÓN ADICIONAL: Capacidad de pago basada en ingreso
        # Máximo 3x el ingreso mensual (regla bancaria conservadora)
        limite_ingreso = ingreso * 3.0
        
        if credito_base > limite_ingreso:
            credito_final = limite_ingreso
        else:
            credito_final = credito_base
        
        return credito_final

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
