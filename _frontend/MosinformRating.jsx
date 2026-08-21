import { useCallback, useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import Cookies from 'js-cookie';

import Content from '@/components/content/Content';
import Layout from '@/components/layout/Layout';
import Button from '@/components/ui/button/Button';
import FileUploader from '@/components/ui/file-uploader/FileUploader';
import LeftMenu from '@/components/ui/left-menu/LeftMenu';
import LeftMenuActive from '@/components/ui/left-menu/left-menu-active/LeftMenuActive';

import { $axios } from '@/api';
import { API_URL, TOKEN } from '@/app.constants';
import { useCheckAuth } from '@/hooks/useCheckAuth';

import styles from './MosinformRating.module.scss';

const MosinformRating = () => {
	useCheckAuth();
	const { active_menu } = useSelector(store => store.booleanValues);

	const [files, setFiles] = useState([]);
	const [period, setPeriod] = useState('');
	const [jobId, setJobId] = useState(null);
	const [status, setStatus] = useState(null);
	const [error, setError] = useState('');
	const [busy, setBusy] = useState(false);
	const timer = useRef(null);

	useEffect(() => {
		return () => {
			if (timer.current) clearInterval(timer.current);
		};
	}, []);

	const poll = useCallback(async id => {
		try {
			const { data } = await $axios.get(`/mosinform/jobs/${id}`);
			setStatus(data);
			if (data.status === 'done' || data.status === 'error') {
				setBusy(false);
				if (timer.current) clearInterval(timer.current);
			}
		} catch (e) {
			setError(e.response?.data?.detail || e.message);
			setBusy(false);
		}
	}, []);

	const start = async () => {
		if (!files.length) {
			setError('Загрузите Word/Excel Медиалогии или zip');
			return;
		}
		setError('');
		setBusy(true);
		setStatus(null);
		const form = new FormData();
		form.append('period', period);
		files.forEach(f => form.append('files', f));
		try {
			const token = Cookies.get(TOKEN);
			const res = await fetch(`${API_URL}/mosinform/jobs`, {
				method: 'POST',
				headers: token ? { Authorization: `Bearer ${token}` } : {},
				body: form,
			});
			if (!res.ok) {
				const detail = await res.text();
				throw new Error(detail || res.statusText);
			}
			const data = await res.json();
			setJobId(data.job_id);
			if (timer.current) clearInterval(timer.current);
			timer.current = setInterval(() => poll(data.job_id), 2500);
			poll(data.job_id);
		} catch (e) {
			setError(e.message);
			setBusy(false);
		}
	};

	const download = kind => {
		if (!jobId) return;
		const token = Cookies.get(TOKEN);
		fetch(`${API_URL}/mosinform/jobs/${jobId}/${kind}`, {
			headers: token ? { Authorization: `Bearer ${token}` } : {},
		})
			.then(r => r.blob())
			.then(blob => {
				const a = document.createElement('a');
				a.href = URL.createObjectURL(blob);
				a.download = `mosinform_rating.${kind === 'pptx' ? 'pptx' : 'xlsx'}`;
				a.click();
			});
	};

	return (
		<Layout>
			{active_menu ? <LeftMenuActive /> : <LeftMenu />}
			<Content>
				<div className={styles.page}>
					<h1 className={styles.title}>Мосинформ.Рейтинг</h1>
					<p className={styles.lead}>
						Загрузите выгрузки Медиалогии (docx, xlsx или zip). Система разберёт тексты,
						разметит объекты локальной моделью и соберёт презентацию.
					</p>
					<label className={styles.label}>Период на слайдах</label>
					<input
						className={styles.input}
						placeholder="Июль 2026"
						value={period}
						onChange={e => setPeriod(e.target.value)}
						disabled={busy}
					/>
					<FileUploader
						acceptedTypes=".docx,.xlsx,.xls,.zip"
						onFileUpload={next => setFiles(prev => [...prev, ...next])}
					/>
					{files.length > 0 && (
						<ul className={styles.files}>
							{files.map(f => (
								<li key={f.name}>
									{f.name} · {(f.size / 1024 / 1024).toFixed(1)} МБ
								</li>
							))}
						</ul>
					)}
					<div className={styles.actions}>
						<Button onClick={start} disabled={busy}>
							{busy ? 'Считаем…' : 'Собрать презентацию'}
						</Button>
						{files.length > 0 && !busy && (
							<button className={styles.clear} type="button" onClick={() => setFiles([])}>
								Очистить файлы
							</button>
						)}
					</div>
					{error && <div className={styles.error}>{error}</div>}
					{status && (
						<div className={styles.status}>
							<div>
								<strong>{status.status}</strong> — {status.message}
							</div>
							{status.summary?.messages != null && (
								<p>
									Текстов: {status.summary.messages}, объектов:{' '}
									{status.summary.objects}, без объекта: {status.summary.untagged}
								</p>
							)}
							{status.has_pptx && (
								<div className={styles.actions}>
									<Button onClick={() => download('pptx')}>Скачать PPTX</Button>
									{status.has_xlsx && (
										<Button onClick={() => download('xlsx')}>Скачать Excel</Button>
									)}
								</div>
							)}
						</div>
					)}
				</div>
			</Content>
		</Layout>
	);
};

export default MosinformRating;
