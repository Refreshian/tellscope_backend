import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import Cookies from 'js-cookie';

import { $axios } from '@/api';
import { API_URL, TOKEN } from '@/app.constants';

import styles from './MosinformArchive.module.scss';

const ARCHIVE_ID = '__archive__';
const STATUS_LABEL = {
	queued: 'В очереди',
	running: 'Считается',
	done: 'Готово',
	error: 'Ошибка',
	unknown: 'Неизвестно',
};

const formatWhen = iso => {
	if (!iso || String(iso).startsWith('2020-01-01')) return '—';
	const date = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T'));
	if (Number.isNaN(date.getTime())) return iso;
	return date.toLocaleString('ru-RU');
};

const downloadJob = (jobId, kind) => {
	const token = Cookies.get(TOKEN);
	fetch(`${API_URL}/mosinform/jobs/${jobId}/${kind}`, {
		headers: token ? { Authorization: `Bearer ${token}` } : {},
	})
		.then(response => {
			if (!response.ok) throw new Error('Файл ещё не готов');
			return response.blob();
		})
		.then(blob => {
			const link = document.createElement('a');
			link.href = URL.createObjectURL(blob);
			link.download = `mosinform_rating_${jobId}.${kind}`;
			link.click();
		})
		.catch(() => {});
};

const fetchJobs = async () => {
	try {
		const { data } = await $axios.get('/mosinform/jobs');
		if (Array.isArray(data?.jobs)) {
			return data.jobs.filter(job => job.job_id && job.job_id !== ARCHIVE_ID);
		}
	} catch {
		// старый бэкенд ещё без GET /mosinform/jobs — читаем сводку
	}
	const { data } = await $axios.get(`/mosinform/jobs/${ARCHIVE_ID}`);
	const notes = data?.summary?.notes;
	if (Array.isArray(notes)) return notes.filter(job => job.job_id && job.job_id !== ARCHIVE_ID);
	if (typeof notes === 'string') {
		try {
			const parsed = JSON.parse(notes);
			return Array.isArray(parsed)
				? parsed.filter(job => job.job_id && job.job_id !== ARCHIVE_ID)
				: [];
		} catch {
			return [];
		}
	}
	return [];
};

const MosinformArchive = ({ filterText = '' }) => {
	const [jobs, setJobs] = useState([]);
	const [error, setError] = useState('');
	const [loading, setLoading] = useState(true);

	const load = useCallback(async () => {
		try {
			const next = await fetchJobs();
			setJobs(next);
			setError('');
		} catch (e) {
			setError(e.response?.data?.detail || e.message || 'Не удалось загрузить расчёты');
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		load();
		const timer = setInterval(load, 4000);
		return () => clearInterval(timer);
	}, [load]);

	const visible = useMemo(() => {
		const query = filterText.trim().toLowerCase();
		if (!query) return jobs;
		return jobs.filter(job =>
			[job.period, job.files, job.status, job.message, job.job_id]
				.filter(Boolean)
				.join(' ')
				.toLowerCase()
				.includes(query),
		);
	}, [jobs, filterText]);

	if (loading) {
		return <p className={styles.hint}>Загружаем сохранённые расчёты…</p>;
	}

	if (error && !jobs.length) {
		return <p className={styles.errorBox}>{error}</p>;
	}

	if (!visible.length) {
		return (
			<div className={styles.empty}>
				<p>Пока нет сохранённых расчётов Мосинформ.Рейтинг.</p>
				<Link className={styles.link} to="/mosinform-rating">
					Запустить расчёт
				</Link>
			</div>
		);
	}

	return (
		<div className={styles.list}>
			<div className={styles.head}>
				<p className={styles.lead}>
					Результаты сохраняются на сервере. Можно закрыть браузер и вернуться сюда.
				</p>
				<Link className={styles.link} to="/mosinform-rating">
					Новый расчёт
				</Link>
			</div>
			{visible.map(job => (
				<article key={job.job_id} className={styles.card}>
					<div className={styles.row}>
						<span className={`${styles.badge} ${styles[job.status] || ''}`}>
							{STATUS_LABEL[job.status] || job.status}
						</span>
						<strong>{job.period || 'Период не указан'}</strong>
					</div>
					<p>
						<span>Файлы:</span> {job.files || '—'}
					</p>
					<p>
						<span>Статус:</span> {job.message || '—'}
					</p>
					{job.summary?.messages != null && (
						<p>
							<span>Итог:</span> текстов {job.summary.messages}, объектов{' '}
							{job.summary.objects}, без объекта {job.summary.untagged}
						</p>
					)}
					{(job.lineage?.model_id || job.lineage?.prompt_id) && (
						<p className={styles.meta}>
							{job.lineage.model_id || ''}
							{job.lineage.prompt_id ? ` · ${job.lineage.prompt_id}` : ''}
							{job.lineage.git_sha ? ` · ${job.lineage.git_sha}` : ''}
						</p>
					)}
					{job.stale && (
						<p className={styles.meta}>Задача не обновлялась больше двух часов</p>
					)}
					<p className={styles.meta}>
						{formatWhen(job.created_at)}
						{job.updated_at ? ` · обновлено ${formatWhen(job.updated_at)}` : ''}
					</p>
					{(job.has_pptx || job.has_xlsx) && (
						<div className={styles.actions}>
							{job.has_pptx && (
								<button type="button" onClick={() => downloadJob(job.job_id, 'pptx')}>
									Скачать PPTX
								</button>
							)}
							{job.has_xlsx && (
								<button type="button" onClick={() => downloadJob(job.job_id, 'xlsx')}>
									Скачать Excel
								</button>
							)}
						</div>
					)}
				</article>
			))}
		</div>
	);
};

export default MosinformArchive;
